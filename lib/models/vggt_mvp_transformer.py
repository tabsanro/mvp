# Copyright 2021 Garena Online Private Limited.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Integration of VGGT Aggregator with MVPPose transformer
# Modified from the original MVPPose implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from models import pose_resnet
from models.vggt_mvp_decoder import MvPDecoderLayer, MvPDecoder
from models.vggt_aggregator import Aggregator
from models.matcher import HungarianMatcher
from core.loss import PerJointL1Loss, PerBoneL1Loss, PerProjectionL1Loss
from models.util.misc import (
    accuracy, get_world_size,
    is_dist_avail_and_initialized, inverse_sigmoid)
from models.position_encoding import PositionEmbeddingSine, get_rays_new, get_2d_coords
from torch.nn.init import xavier_uniform_, constant_, normal_


def sigmoid_focal_loss(inputs, targets, num_samples,
                       alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection:
    https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_samples


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h,
                                            h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VGGTAdapter(nn.Module):
    """
    Adapter to connect VGGT Aggregator outputs to MVPPose decoder inputs
    Converts VGGT patch tokens to multi-scale backbone features and camera tokens to camera rays
    """
    def __init__(self, vggt_dim, mvp_dim, num_views, patch_start_idx, image_size, num_feat_levels=3, patch_size=14):
        super().__init__()
        self.vggt_dim = vggt_dim
        self.mvp_dim = mvp_dim
        self.num_views = num_views
        self.patch_start_idx = patch_start_idx
        self.image_size = image_size
        self.num_feat_levels = num_feat_levels
        self.patch_size = patch_size

        # Projection layer for patch tokens to match dimensions
        self.patch_proj = nn.Linear(vggt_dim * 2, mvp_dim)  # *2 because of concat
        
        # Camera token projection to generate camera rays
        self.camera_proj = nn.Sequential(
            nn.Linear(vggt_dim * 2, mvp_dim),
            nn.ReLU(),
            nn.Linear(mvp_dim, mvp_dim)  # For 3D ray directions
        )
        
        # Multi-scale feature generation from patch tokens
        self.multiscale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vggt_dim * 2, mvp_dim),
                nn.ReLU(),
                nn.Linear(mvp_dim, mvp_dim)
            ) for _ in range(num_feat_levels)
        ])
        
        # Learnable view embeddings for multi-view fusion
        self.view_embed = nn.Parameter(torch.randn(num_views, mvp_dim))
        
        # Spatial reshaping layers for different scales
        self.patch_size = 14  # VGGT patch size
        
    def forward(self, vggt_outputs, camera_tokens, patch_tokens, patch_start_idx):
        """
        Convert VGGT aggregator outputs to MVPPose format
        
        Args:
            vggt_outputs: List of tensors [B, S, P, 2*C] from VGGT (for compatibility)
            camera_tokens: List of tensors [B, S, 1, 2*C] - camera tokens from each layer
            patch_tokens: List of tensors [B, S, P_patches, 2*C] - patch tokens from each layer
            patch_start_idx: Index where patch tokens start
            
        Returns:
            backbone_features: List of multi-scale features for decoder
            camera_rays: List of camera ray features
            spatial_shapes: Spatial dimensions for each scale
        """
        # Use the last layer outputs
        last_camera_tokens = camera_tokens[-1]  # [B, S, 1, 2*C]
        last_patch_tokens = patch_tokens[-1]    # [B, S, P_patches, 2*C]
        
        B, S, P_patches, feat_dim = last_patch_tokens.shape          
        
        # Generate multi-scale backbone features from patch tokens
        backbone_features = []
        spatial_shapes = []
        
        # Calculate patch grid dimensions from image size and patch size
        # image_size expected as [height, width] or single int for square
        if isinstance(self.image_size, (list, tuple)):
            img_h, img_w = int(self.image_size[0]), int(self.image_size[1])
        elif hasattr(self.image_size, 'shape'):
            # torch tensor or numpy
            img_h, img_w = int(self.image_size[0]), int(self.image_size[1])
        else:
            img_h = img_w = int(self.image_size)

        patch_h = max(1, img_h // self.patch_size)
        patch_w = max(1, img_w // self.patch_size)
        # Sanity: ensure product matches P_patches when possible
        if patch_h * patch_w != P_patches:
            # fallback: try to infer grid by nearest factors (keep original P_patches)
            # but prefer computed patch_h/patch_w to avoid sqrt assumptions
            # we'll proceed with computed patch_h/patch_w and adjust indices when needed
            pass
        
        for level in range(self.num_feat_levels):
            # Project patch tokens for this scale
            level_features = self.multiscale_projs[level](last_patch_tokens)  # [B, S, P_patches, mvp_dim]
            
            # Calculate spatial dimensions for this level
            scale_factor = 2 ** level
            level_h = max(1, patch_h // scale_factor)
            level_w = max(1, patch_w // scale_factor)
            spatial_shapes.append((level_h, level_w))
            
            # Subsample patches for this scale (simple approach)
            if scale_factor > 1:
                # Take evenly spaced indices and ensure we take exactly level_h*level_w
                step = max(1, P_patches // (level_h * level_w))
                indices = torch.arange(0, P_patches, step).long()[:level_h * level_w]
                level_features = level_features[:, :, indices, :]  # [B, S, P_level, mvp_dim]
            
            # Reshape to match expected format [B*S, mvp_dim, H, W]
            # If the number of patches doesn't exactly match level_h*level_w, try to reshape
            P_level = level_features.shape[2]
            expected = level_h * level_w
            if P_level != expected:
                # If possible, try to reshape by padding/truncating
                if P_level > expected:
                    level_features = level_features[:, :, :expected, :]
                else:
                    # pad with zeros
                    pad = torch.zeros((B, S, expected - P_level, self.mvp_dim), device=level_features.device, dtype=level_features.dtype)
                    level_features = torch.cat([level_features, pad], dim=2)

            level_features = level_features.view(B * S, level_h, level_w, self.mvp_dim)
            level_features = level_features.permute(0, 3, 1, 2)  # [B*S, mvp_dim, H, W]
            
            backbone_features.append(level_features)
        
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=last_patch_tokens.device)
        
        return backbone_features, spatial_shapes


class VGGTMVPPoseTransformer(nn.Module):
    """
    Integrated model combining VGGT Aggregator with MVPPose transformer
    """
    def __init__(self, backbone, cfg):
        super(VGGTMVPPoseTransformer, self).__init__()
        
        # Basic MVPPose settings
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_instance = cfg.DECODER.num_instance
        self.backbone = backbone
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET
        
        # VGGT Aggregator
        self.vggt_aggregator = Aggregator(
            img_size=cfg.NETWORK.IMAGE_SIZE[0],  # Assuming square images
            patch_size=getattr(cfg, 'VGGT', {}).get('PATCH_SIZE', 14),
            embed_dim=getattr(cfg, 'VGGT', {}).get('EMBED_DIM', 768),
            depth=getattr(cfg, 'VGGT', {}).get('DEPTH', 12),
            num_heads=getattr(cfg, 'VGGT', {}).get('NUM_HEADS', 12),
            mlp_ratio=getattr(cfg, 'VGGT', {}).get('MLP_RATIO', 4.0),
            num_register_tokens=getattr(cfg, 'VGGT', {}).get('NUM_REGISTER_TOKENS', 4),
            aa_order=getattr(cfg, 'VGGT', {}).get('AA_ORDER', ["frame", "global"]),
            rope_freq=getattr(cfg, 'VGGT', {}).get('ROPE_FREQ', 100),
            qkv_bias=getattr(cfg, 'VGGT', {}).get('QKV_BIAS', True),
            proj_bias=getattr(cfg, 'VGGT', {}).get('PROJ_BIAS', True),
            ffn_bias=getattr(cfg, 'VGGT', {}).get('FFN_BIAS', True),
            qk_norm=getattr(cfg, 'VGGT', {}).get('QK_NORM', True),
            init_values=getattr(cfg, 'VGGT', {}).get('INIT_VALUES', 0.01),
            patch_embed=getattr(cfg, 'VGGT', {}).get('PATCH_EMBED', "conv"),
        )
        
        # Adapter to connect VGGT to MVPPose
        self.vggt_adapter = VGGTAdapter(
            vggt_dim=getattr(cfg, 'VGGT', {}).get('EMBED_DIM', 768),
            mvp_dim=cfg.DECODER.d_model,
            num_views=cfg.DATASET.CAMERA_NUM,
            patch_start_idx=self.vggt_aggregator.patch_start_idx,
            image_size=cfg.NETWORK.IMAGE_SIZE,
            patch_size=getattr(cfg, 'VGGT', {}).get('PATCH_SIZE', 14),
            num_feat_levels=cfg.DECODER.num_feature_levels
        )
        
        # MVPPose components
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)

        self.reference_points = nn.Linear(cfg.DECODER.d_model, 3)
        self.reference_feats = nn.Linear(
            cfg.DECODER.d_model * len(cfg.DECODER.use_feat_level) * cfg.DATASET.CAMERA_NUM,
            cfg.DECODER.d_model)

        # MVPPose decoder
        decoder_layer = MvPDecoderLayer(
            cfg.MULTI_PERSON.SPACE_SIZE,
            cfg.MULTI_PERSON.SPACE_CENTER,
            cfg.NETWORK.IMAGE_SIZE,
            cfg.DECODER.d_model,
            cfg.DECODER.dim_feedforward,
            cfg.DECODER.dropout,
            cfg.DECODER.activation,
            cfg.DECODER.num_feature_levels,
            cfg.DECODER.nhead,
            cfg.DECODER.dec_n_points,
            cfg.DECODER.detach_refpoints_cameraprj_firstlayer,
            cfg.DECODER.fuse_view_feats,
            cfg.DATASET.CAMERA_NUM,
            cfg.DECODER.projattn_posembed_mode
        )
        
        self.decoder = MvPDecoder(
            cfg, decoder_layer,
            cfg.DECODER.num_decoder_layers,
            cfg.DECODER.return_intermediate_dec
        )

        # Query embeddings
        num_queries = cfg.DECODER.num_instance * cfg.DECODER.num_keypoints
        self.query_embed_type = cfg.DECODER.query_embed_type
        
        if self.query_embed_type == 'person_joint':
            self.joint_embedding = nn.Embedding(cfg.DECODER.num_keypoints, cfg.DECODER.d_model * 2)
            self.instance_embedding = nn.Embedding(cfg.DECODER.num_instance, cfg.DECODER.d_model * 2)
        elif self.query_embed_type == 'per_joint':
            self.query_embed = nn.Embedding(num_queries, cfg.DECODER.d_model * 2)

        # Position encoding and view embeddings
        N_steps = cfg.DECODER.d_model // 2
        self.pos_encoding = PositionEmbeddingSine(N_steps, normalize=True)
        self.view_embed = nn.Parameter(torch.Tensor(cfg.DATASET.CAMERA_NUM, cfg.DECODER.d_model))
        
        # Output heads
        num_pred = self.decoder.num_layers
        num_classes = 2
        
        self.class_embed = nn.Linear(cfg.DECODER.d_model, num_classes)
        self.pose_embed = MLP(cfg.DECODER.d_model, cfg.DECODER.d_model, 3, cfg.DECODER.pose_embed_layer)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Loss and matching
        if cfg.DECODER.with_pose_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.pose_embed = _get_clones(self.pose_embed, num_pred)
            self.decoder.pose_embed = self.pose_embed
        else:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.pose_embed = nn.ModuleList([self.pose_embed for _ in range(num_pred)])
            self.decoder.pose_embed = None

        # Store config parameters
        self.pred_conf_threshold = cfg.DECODER.pred_conf_threshold
        self.pred_class_fuse = cfg.DECODER.pred_class_fuse
        self.aux_loss = cfg.DECODER.aux_loss
        self.use_feat_level = cfg.DECODER.use_feat_level
        self.query_adaptation = cfg.DECODER.query_adaptation
        self.convert_joint_format_indices = cfg.DECODER.convert_joint_format_indices
        
        # Camera token adaptation projection
        if self.query_adaptation:
            vggt_embed_dim = getattr(cfg, 'VGGT', {}).get('EMBED_DIM', 768)
            self.camera_adapt_proj = nn.Linear(vggt_embed_dim * 2, cfg.DECODER.d_model)
        
        # Setup criterion
        matcher = HungarianMatcher(
            match_coord=cfg.DECODER.match_coord,
            cost_class=2.,
            cost_pose=5.
        )
        
        weight_dict = {
            'loss_ce': cfg.DECODER.loss_weight_loss_ce,
            'loss_pose_perjoint': cfg.DECODER.loss_pose_perjoint,
            'loss_pose_perbone': cfg.DECODER.loss_pose_perbone
        }
        
        losses = ['joints', 'labels', 'cardinality']
        
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(num_pred - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        from models.multi_view_pose_transformer import SetCriterion  # Import from original
        self.criterion = SetCriterion(
            num_classes, matcher, weight_dict, losses, cfg,
            focal_alpha=0.25, root_idx=self.root_id
        )
        
        device = torch.device('cuda')
        self.criterion.to(device)

    def _reset_parameters(self):
        """Initialize parameters"""
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.view_embed)
        
        # Initialize output heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(2) * bias_value
        nn.init.constant_(self.pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.pose_embed.layers[-1].bias.data, 0)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def absolute2norm(self, absolute_coords):
        device = absolute_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        norm_coords = (absolute_coords - grid_center + grid_size / 2.0) / grid_size
        return norm_coords

    def norm2absolute(self, norm_coords):
        device = norm_coords.device
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = norm_coords * grid_size + grid_center - grid_size / 2.0
        return loc

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_poses': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def forward(self, views=None, meta=None):
        batch, _, imageh, imagew = views[0].shape
        nview = len(views)
        
        # Process images with VGGT Aggregator
        # Stack views to create sequence dimension
        images = torch.stack(views, dim=1)  # [B, S, C, H, W]
        
        # Get VGGT features - now returns separated tokens
        vggt_outputs, camera_tokens, patch_tokens, patch_start_idx = self.vggt_aggregator(images)
        
        # Convert VGGT outputs to MVPPose format (backbone features and camera rays)
        backbone_feats, vggt_spatial_shapes = self.vggt_adapter(
            vggt_outputs, camera_tokens, patch_tokens, patch_start_idx
        )
        
        # No need for backbone processing - using VGGT features directly
        # No need for explicit camera calibration - using VGGT camera tokens
        
        nfeat_level = len(backbone_feats)

        # Prepare features for decoder using VGGT outputs
        src_flatten_views = []
        mask_flatten_views = []
        spatial_shapes_views = []

        for lvl, src in enumerate(backbone_feats):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            mask = src.new_zeros(bs, h, w).bool()
            mask_flatten_views.append(mask)
            src_flatten_views.append(src)

        # Use VGGT spatial shapes if available, otherwise compute from backbone features
        if len(spatial_shapes_views) > 0:
            spatial_shapes_views = torch.as_tensor(spatial_shapes_views, dtype=torch.long, device=images.device)
        else:
            spatial_shapes_views = vggt_spatial_shapes
            
        level_start_index_views = torch.cat((
            torch.zeros((1, ), dtype=torch.long, device=images.device),
            spatial_shapes_views.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios_views = torch.stack([self.get_valid_ratio(m) for m in mask_flatten_views], 1)
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]

        # Query embeddings
        if self.query_embed_type == 'person_joint':
            joint_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (joint_embeds + instance_embeds).flatten(0, 1)
        elif self.query_embed_type == 'per_joint':
            query_embeds = self.query_embed.weight

        query_embed, tgt = torch.split(query_embeds, query_embeds.shape[1] // 2, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch, -1, -1)

        # Reference points
        if self.query_adaptation:
            # Use VGGT camera tokens for query adaptation
            last_camera_tokens = camera_tokens[-1]  # [B, S, 1, 2*C]
            # Average across views and project to mvp_dim
            cam_adapt_feats = last_camera_tokens.mean(dim=1).squeeze(1)  # [B, 2*C]
            cam_adapt_feats = self.camera_adapt_proj(cam_adapt_feats)  # [B, mvp_dim]
            ref_feats = cam_adapt_feats.unsqueeze(1)  # [B, 1, mvp_dim]
            reference_points = self.reference_points(query_embed + ref_feats).sigmoid()
        else:
            reference_points = self.reference_points(query_embed).sigmoid()

        init_reference = reference_points

        # Decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, src_flatten_views,
            meta=meta, src_spatial_shapes=spatial_shapes_views,
            src_level_start_index=level_start_index_views,
            src_valid_ratios=valid_ratios_views,
            query_pos=query_embed,
            src_padding_mask=mask_flatten_views
        )

        # Output processing
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            if self.pred_class_fuse == 'mean':
                outputs_class = self.class_embed[lvl](hs[lvl]).\
                    view(batch, self.num_instance, self.num_joints, -1).\
                    sigmoid().mean(2)
                outputs_class = inverse_sigmoid(outputs_class)
            elif self.pred_class_fuse == 'feat_mean_pool':
                outputs_class = self.class_embed[lvl](hs[lvl])\
                    .view(batch, self.num_instance, self.num_joints, -1)\
                    .mean(2)
            else:
                outputs_class = self.class_embed[lvl](
                    hs[lvl].view(batch, self.num_instance, self.num_joints, -1).max(2)[0]
                )
            
            tmp = self.pose_embed[lvl](hs[lvl])
            tmp += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            
            if self.convert_joint_format_indices is not None:
                outputs_coord = outputs_coord.view(batch, self.num_instance, self.num_joints, -1)
                outputs_coord = outputs_coord[..., self.convert_joint_format_indices, :]
                outputs_coord = outputs_coord.flatten(1, 2)

            outputs_coords.append({'outputs_coord': outputs_coord})

        out = {'pred_logits': outputs_classes[-1],
               'pred_poses': outputs_coords[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes, outputs_coords)

        if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
            meta[0]['roots_3d_norm'] = self.absolute2norm(meta[0]['roots_3d'].float())
            meta[0]['joints_3d_norm'] = self.absolute2norm(meta[0]['joints_3d'].float())
            loss_dict = self.criterion(out, meta)
            return out, loss_dict

        return out


def get_vggt_mvp(cfg, is_train=True):
    """
    Factory function to create VGGT-MVPPose integrated model
    """
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    
    model = VGGTMVPPoseTransformer(backbone, cfg)
    return model