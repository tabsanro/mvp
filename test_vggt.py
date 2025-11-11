import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import the model class (file under test)
from models.vggt_mvp_transformer import VGGTMVPPoseTransformer, inverse_sigmoid

# Fake decoder to mimic expected decoder outputs
class FakeDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_queries):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_queries = num_queries

    def forward(self, tgt, reference_points, src_flatten_views, camera_rays,
                meta=None, src_spatial_shapes=None, src_level_start_index=None,
                src_valid_ratios=None, query_pos=None, src_padding_mask=None):
        batch = tgt.shape[0]
        # hs shape: [num_layers, batch, num_queries, d_model]
        hs = torch.zeros(self.num_layers, batch, self.num_queries, self.d_model, dtype=tgt.dtype, device=tgt.device)
        # make inter_references list length num_layers-1
        inter_references = [reference_points for _ in range(max(0, self.num_layers - 1))]
        return hs, inter_references

# Minimal forward test by constructing a model object without running its __init__
def make_minimal_model(batch=2, nview=3, d_model=64, num_instance=2, num_joints=5, img_h=224, img_w=224):
    num_queries = num_instance * num_joints

    # create bare object
    model = object.__new__(VGGTMVPPoseTransformer)
    # initialize nn.Module base so we can assign submodules safely
    nn.Module.__init__(model)

    # meta / basic attrs used in forward
    model.num_joints = num_joints
    model.num_instance = num_instance
    model.image_size = (img_h, img_w)
    model.root_id = 0
    model.dataset_name = "dummy"
    model.pred_class_fuse = 'other'  # use else branch in forward
    model.convert_joint_format_indices = None
    model.aux_loss = False
    model.query_embed_type = 'per_joint'
    model.query_adaptation = False

    # reference_points linear: maps d_model -> 3
    model.reference_points = nn.Linear(d_model, 3)

    # simple class & pose heads per decoder layer (we'll set num_layers=2)
    num_layers = 2
    model.class_embed = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(num_layers)])
    # pose_embed should accept (batch*num_queries, d_model) or be applied per-query: use Linear
    model.pose_embed = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])

    # query embedding (per_joint)
    num_queries = num_instance * num_joints
    # provide a tensor of shape (num_queries, d_model*2)
    q = torch.randn(num_queries, d_model * 2)
    model.query_embed = q

    # adapter & aggregator: produce backbone_feats and camera_rays
    def fake_aggregator(images):
        # returns vggt_outputs, camera_tokens, patch_tokens, patch_start_idx
        B, S, C, H, W = images.shape
        # create dummy camera tokens list (only last used in code by adapter if it were used)
        camera_tokens = [torch.randn(B, S, 1, 2 * 32)]
        # patch_tokens unused by our adapter replaced below
        patch_tokens = [torch.randn(B, S, 16, 2 * 32)]
        vggt_outputs = None
        patch_start_idx = 0
        return vggt_outputs, camera_tokens, patch_tokens, patch_start_idx

    # Adapter returns multi-scale backbone features and camera_rays
    def fake_adapter(vggt_outputs, camera_tokens, patch_tokens, patch_start_idx):
        B = batch
        S = nview
        # create two feature levels with small H/W
        level1 = torch.randn(B * S, d_model, 14, 14)
        level2 = torch.randn(B * S, d_model, 7, 7)
        backbone_feats = [level1, level2]
        # camera_rays: list of tensors per level (B*S, P_level, 3)
        cam_rays = [torch.randn(B * S, 14 * 14, 3), torch.randn(B * S, 7 * 7, 3)]
        spatial_shapes = torch.tensor([[14, 14], [7, 7]], dtype=torch.long)
        return backbone_feats, cam_rays, spatial_shapes

    model.vggt_aggregator = fake_aggregator
    model.vggt_adapter = fake_adapter

    # decoder
    model.decoder = FakeDecoder(num_layers, d_model, num_queries)

    # bind get_valid_ratio method from class (unbound function will be bound automatically)
    # grid_size/center still not needed for forward path here
    model.grid_size = torch.tensor([2000., 2000., 2000.])
    model.grid_center = torch.tensor([0., 0., 1000.])

    return model

def run_forward_test():
    torch.manual_seed(0)
    batch = 2
    nview = 3
    d_model = 64
    num_instance = 2
    num_joints = 5
    img_h = img_w = 224

    model = make_minimal_model(batch=batch, nview=nview, d_model=d_model,
                               num_instance=num_instance, num_joints=num_joints,
                               img_h=img_h, img_w=img_w)

    # Create dummy multi-view inputs: list of S tensors each [B, C, H, W]
    views = [torch.randn(batch, 3, img_h, img_w) for _ in range(nview)]
    meta = [{} for _ in range(batch)]

    # Run forward
    out = model.forward(views=views, meta=meta)

    print("Forward output keys:", list(out.keys()))
    if 'pred_logits' in out:
        print("pred_logits shape:", out['pred_logits'].shape)
    if 'pred_poses' in out:
        coords = out['pred_poses']['outputs_coord']
        print("pred_poses outputs_coord shape:", coords.shape)

if __name__ == "__main__":
    run_forward_test()