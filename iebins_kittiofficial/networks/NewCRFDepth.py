import torch
import torch.nn as nn
import torch.nn.functional as F

from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .depth_update  import *
from .swin_transformer_v2 import SwinTransformerV2
########################################################################################################################


class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)

        in_channels = [192, 384, 768, 1536]
        self.backbone = SwinTransformerV2(
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=[22, 22, 22, 11],
            pretrain_window_size=[12, 12, 12, 6],
            drop_path_rate=0.5,
            use_checkpoint=False,
            use_shift=[True, True, False, False],
        )
        
        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(v_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_num = 16
        self.hidden_dim = 128
        self.project = Projection(v_dims[0], self.hidden_dim)

        self.init_weights(pretrained=pretrained)

        self.update = BasicUpdateBlockDepth()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, C, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, C, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, C, 4*H, 4*W)

    def forward(self, imgs, epoch=1, step=100):

        feats = self.backbone(imgs)
        ppm_out = self.decoder(feats)

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        
        # iterative bins
        if epoch == 0 and step < 80:
            max_tree_depth = 3
        else:
            max_tree_depth = 6
        
        if self.up_mode == 'mask':
            mask = self.mask_head(e1)

        b, c, h, w = e1.shape
        device = e1.device
               
        depth = torch.zeros([b, 1, h, w]).to(device)
        context = feats[0]
        gru_hidden = torch.tanh(self.project(e1))
        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.update(depth, context, gru_hidden, max_tree_depth, self.depth_num, self.min_depth, self.max_depth)
        
        if self.up_mode == 'mask':
            for i in range(len(pred_depths_r_list)):
                pred_depths_r_list[i] = self.upsample_mask(pred_depths_r_list[i], mask)  
            for i in range(len(pred_depths_c_list)):
                pred_depths_c_list[i] = self.upsample_mask(pred_depths_c_list[i], mask.detach())
            for i in range(len(uncertainty_maps_list)):
                uncertainty_maps_list[i] = self.upsample_mask(uncertainty_maps_list[i], mask.detach())                   
        else:
            for i in range(len(pred_depths_r_list)):
                pred_depths_r_list[i] = upsample(pred_depths_r_list[i], scale_factor=4)
            for i in range(len(pred_depths_c_list)):
                pred_depths_c_list[i] = upsample(pred_depths_c_list[i], scale_factor=4) 
            for i in range(len(uncertainty_maps_list)):
                uncertainty_maps_list[i] = upsample(uncertainty_maps_list[i], scale_factor=4) 

        return pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list

class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=192):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim * 2)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.p_head = PHead(hidden_dim, hidden_dim)

    def forward(self, depth, context, gru_hidden, seq_len, depth_num, min_depth, max_depth):
 
        pred_depths_r_list = []
        pred_depths_c_list = []
        uncertainty_maps_list = []
      
        b, _, h, w = depth.shape
        depth_range = max_depth - min_depth
        interval = depth_range / depth_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        index_iter = 0

        for i in range(seq_len):
            input_features = self.encoder(current_depths.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            pred_prob = self.p_head(gru_hidden)

            depth_r = (pred_prob * current_depths.detach()).sum(1, keepdim=True)
            pred_depths_r_list.append(depth_r)

            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_r.repeat(1, depth_num, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_list.append(uncertainty_map)
    
            index_iter = index_iter + 1

            pred_label = get_label(torch.squeeze(depth_r, 1), bin_edges, depth_num).unsqueeze(1)
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())
            pred_depths_c_list.append(depth_c)
           
            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_r.detach(), pred_label.detach(), depth_num, min_depth, max_depth, uncertainty_map)


        return pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list

class PHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(PHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
    
    def forward(self, x):
        out = torch.softmax(self.conv2(F.relu(self.conv1(x))), 1)
        return out

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    def __init__(self, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs 
        self.convd1 = nn.Conv2d(16, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd4 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)
        
    def forward(self, depth):
        d = F.relu(self.convd1(depth))
        d = F.relu(self.convd2(d))
        d = F.relu(self.convd3(d))
        d = F.relu(self.convd4(d))
                
        return d

class Projection(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, 3, padding=1)
        
    def forward(self, x):
        out = self.conv(x)
                
        return out

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def upsample1(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


