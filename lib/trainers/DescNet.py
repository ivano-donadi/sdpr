import torch
import torch.nn as nn
from lib.utils.losses import ArcFace
from lib.networks.layers.partial_fc_v2 import PartialFC_V2

def batched_dist(one, other):
    return 1.0 - torch.nn.functional.cosine_similarity(one, other, dim=1)#(one-other).pow(2).sum(1).sqrt()

class NetworkWrapper(nn.Module):
    def __init__(self, net, margin, descriptor_size, n_clusters, use_ratio, use_arcface):
        super(NetworkWrapper,self).__init__()
        self.net = net
        self.margin = margin
        self.use_ratio = use_ratio
        self.use_arcface = use_arcface
        if self.use_arcface:
            self.arc_crit = ArcFace(s=5, margin = 0.7)
            self.arc_layer = PartialFC_V2(self.arc_crit, descriptor_size, n_clusters)
        else:
            self.triplet_crit = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=1))#nn.TripletMarginLoss(margin=self.margin, p = 2, swap=False)
            self.patch_crit = nn.MSELoss(reduction='mean')

        print("Training setup: use arcface = {0}, use triplet = {1}, use ratio = {2}".format(self.use_arcface, not self.use_arcface, self.use_ratio))            



    def forward(self, batch):


        scalar_stats = {}
        loss = 0

        if not self.use_arcface:
            anchor = self.net(batch['anchor']) # b, d
            # in the end, positive will contain the easiest postive sample
            all_positives = batch['positive'] # b, np, c, h, w
            delta_pos = None
            positive = None
            for i in range(all_positives.shape[1]):
                cpositive = self.net(all_positives[:,i,:,:,:])
                cdelta_pos = batched_dist(anchor,cpositive)
                if delta_pos is None:
                    delta_pos = cdelta_pos
                    positive = cpositive
                else:
                    pos_mask = cdelta_pos < delta_pos #[b, d]
                    positive[pos_mask] = cpositive[pos_mask] #[b, d]
                    delta_pos[pos_mask] = cdelta_pos[pos_mask] #[b,d]

            ap_dist = batched_dist(anchor, positive) #[b,d]

            # in the end, negative will contain the  best semi-hard negative sample
            # the best semi-hard negative sample is the one closest to the anchor but still further away from the anchor than the positive
            # see https://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf section 3.1
            all_negatives = batch['negative'] # b, nn, c, h, w
            delta_neg = None
            negative = None
            for i in range(all_negatives.shape[1]):
                cnegative = self.net(all_negatives[:,i,:,:,:]) # [b,d]
                cdelta_neg = batched_dist(anchor,cnegative)
                if delta_neg is None:
                    delta_neg = cdelta_neg
                    negative = cnegative
                else:
                    neg_mask = (cdelta_neg < delta_neg).logical_and(cdelta_neg > ap_dist) #[b]
                    delta_neg[neg_mask] = cdelta_neg[neg_mask] #[b]
                    negative[neg_mask,:] = cnegative[neg_mask,:] #[b, d]
            
            triplet_loss = self.triplet_crit(anchor, positive, negative)
            scalar_stats.update({'triplet_loss': triplet_loss})


            delta_pos = batched_dist(anchor, positive)
            delta_neg = batched_dist(anchor, negative)

            #regularization_loss = (-torch.log(delta_pos/2)).mean()
            #scalar_stats.update({'regularization_loss': regularization_loss})


            den = delta_pos.exp() + delta_neg.exp()
            ratio_left = (delta_pos.exp() / den)**2
            ratio_right = ( 1 - (delta_neg.exp()/den))**2
            ratio_loss = ratio_left + ratio_right
            ratio_loss = ratio_loss.mean()
            scalar_stats.update({'ratio_loss': ratio_loss})

            loss = loss + triplet_loss #+ 0.5 * regularization_loss
            if self.use_ratio:
                loss = loss + ratio_loss
        else:
            anchor_embedding = self.net(batch['anchor'])
            labels = batch['anchor_label']
            anchor_loss = self.arc_layer(anchor_embedding, labels)
            positive_loss = 0
            for i in range(batch['positive'].shape[1]):
                positive_embedding = self.net(batch['positive'][:,i,:,:,:])
                c_positive_loss = self.arc_layer(positive_embedding, labels)
                positive_loss += c_positive_loss
            positive_loss = positive_loss/batch['positive'].shape[1]
            loss = loss + anchor_loss + positive_loss
            scalar_stats.update({'anchor_loss': anchor_loss})
            scalar_stats.update({'positive_loss': positive_loss})
            scalar_stats.update({'loss': loss})

            anchor = anchor_embedding
            positive = positive_embedding
            negative = self.net(batch['negative'][:,0,:,:,:])
        
        return (anchor, positive, negative), loss, scalar_stats, {} 
