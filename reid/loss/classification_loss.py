import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """Center loss.
    
    Reference: PAST, ICCV 2019
    
    Args:
        num_classes (int): number of classes.
        centers: feature tensors.
    """
    def __init__(self, centers, use_gpu=True):
        super(ClassificationLoss, self).__init__()
        self.classifier = nn.Linear(centers.shape[1], centers.shape[0])
        self.classifier.weight.data.copy_(torch.from_numpy(centers).float())

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.classifier = self.classifier.cuda()

    def forward(self, x, labels, epoch, return_prec=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        output = self.classifier(x)
        labels = labels.long().cuda()
        loss = nn.CrossEntropyLoss()(output, labels)
        if return_prec:
            precision = (output.argmax(1) == labels).sum().float() / labels.size(0)
            return loss, precision
        else:
            return loss
