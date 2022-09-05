#!/usr/bin/python

import copy

import torch as th
from torch import nn



####################
# Helper Functions #
####################
def calculate_feat_map_shape(backbone):
    input_fixture = th.randint(6, (4, 3, 32, 32)).float()
    output = copy.deepcopy(backbone).cpu()(input_fixture)
    shape = tuple(output.shape[1:])

    return shape

#######################
# Intermediate Models #
#######################

class CNNEncoder(nn.Module):
    def __init__(self):
        """Default CNN Model Architecture for Relation Networks Encoder Backbone
        Model Architecture Adapted from Paper for Relation Networks
        """
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

#################
# Model Classes #
#################

class PrototypicalNetworks(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        output_softmax_score: bool = False
    ):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.output_softmax_score = output_softmax_score

    def forward(
        self,
        support_images: th.Tensor,
        support_labels: th.Tensor,
        query_images: th.Tensor,
    ) -> th.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features embeddings  of support and query images
        emb_support = self.backbone.forward(support_images)
        emb_query = self.backbone.forward(query_images)

        # Get number of unique labels in support set
        n_way = len(th.unique(support_labels))


        # Calculate mean embedding vector for each class
        emb_classes_mean_list = []
        for label in range(n_way):
            emb_class_mean = emb_support[th.nonzero(support_labels == label)].mean(0)
            emb_classes_mean_list.append(emb_class_mean)
        
        # Create prototype
        emb_prototype = th.cat(emb_classes_mean_list)

        # Calculate euclidean distance between queries to prototypes
        dists = th.cdist(emb_query, emb_prototype)

        # Convert to classification score
        scores = -dists

        if self.output_softmax_score:
            return nn.Softmax(dim = -1)(scores)
        else:
            return scores

class RelationNetworks(nn.Module):

    def __init__(
        self, 
        backbone: nn.Module, 
        output_softmax_score: bool = False,
        relation_module: nn.Module = None
    ):
        super(RelationNetworks, self).__init__()
        self.backbone = backbone
        self.output_softmax_score = output_softmax_score
        self.feature_dimension = calculate_feat_map_shape(self.backbone)[0]
        self.model_device_location = next(self.backbone.parameters()).device.type
        if relation_module is not None:
            assert isinstance(relation_module, nn.Module), "Please use a proper Torch module"
            self.relation_module_to_use = relation_module.to(self.model_device_location)
        else:
            self.relation_module_to_use = self.relation_module(self.feature_dimension)


    def relation_module(
        self, 
        feature_dimension: int
    ):  
        """ Initiates and returns default CNN for relation module
        """

        # Build the default relation module architecture as defined in paper
        rel_module = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(
                        feature_dimension * 2,
                        feature_dimension,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool2d((5, 5)),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        feature_dimension,
                        feature_dimension,
                        kernel_size=3,
                        padding=0,
                    ),
                    nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool2d((1, 1)),
                ),
                nn.Flatten(),
                nn.Linear(feature_dimension, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
        
        return rel_module.to(self.model_device_location)

    def forward(
        self, 
        support_images: th.Tensor,
        support_labels: th.Tensor,
        query_images: th.Tensor
    ) -> th.Tensor:
        """
        Predict query labels using labeled support images.
        """

        emb_support = self.backbone.forward(support_images)
        emb_query = self.backbone.forward(query_images)

        # Get number of unique labels in support set
        n_way = len(th.unique(support_labels))
        
        # Calculate mean embedding map for each class
        emb_classes_mean_list = []
        for label in range(n_way):
            class_idxes = th.where(support_labels==label)[0].unsqueeze(-1).to(self.model_device_location)
            emb_class_mean = emb_support[class_idxes,].mean(0)
            emb_classes_mean_list.append(emb_class_mean)
        
        emb_prototype = th.cat(emb_classes_mean_list)

        # Concatenate feature maps of each pair of query to prototyper
        query_prototype_feature_pairs = th.cat(
            (
                emb_prototype.unsqueeze(dim=0).expand(
                    emb_query.shape[0], -1, -1, -1, -1
                ),
                emb_query.unsqueeze(dim=1).expand(
                    -1, emb_prototype.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *emb_query.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1] 
        scores = self.relation_module_to_use(query_prototype_feature_pairs)
        
        # reshape to allow relation_scores to be of shape (n_queries, n_prototypes).
        scores= scores.view(
            -1, emb_prototype.shape[0]
        )
        
        if self.output_softmax_score:
            return nn.Softmax(dim=-1)(scores)
        else:
            return scores