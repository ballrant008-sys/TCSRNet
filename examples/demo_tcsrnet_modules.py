"""Sanity check of the openly released modules using synthetic tensors.
Does NOT perform full TCSRNet infrared detection or reproduce paper metrics.
"""
import torch
from model.TCSRNet import LDTR, LGBI, TRHR

def main():
    torch.manual_seed(7)
    c = 32  # illustrative component test, not a full-paper model configuration
    current = torch.randn(1, c, 32, 32)
    deeper = torch.randn(1, c, 16, 16)
    ldtr, lgbi, trhr = LDTR(c).eval(), LGBI(c).eval(), TRHR(c).eval()
    with torch.no_grad():
        enhanced, target_ref = ldtr(current)
        boundary_feature, edge_map = lgbi(enhanced, deeper, target_ref)
        output = trhr(boundary_feature, deeper, target_ref)
    assert enhanced.shape == current.shape
    assert target_ref.shape == (1, 1, 32, 32)
    assert edge_map.shape == (1, 1, 32, 32)
    assert output.shape == current.shape
    print('LDTR, LGBI, TRHR module interfaces: PASS')
    print('Shapes:', enhanced.shape, target_ref.shape, edge_map.shape, output.shape)
    print('This is a partial module demo, not end-to-end TCSRNet inference.')

if __name__=='__main__':
    main()
