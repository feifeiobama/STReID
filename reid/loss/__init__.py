from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss, HoughTripletLoss, RankingTripletLoss, SortedTripletLoss
from .center_loss import CenterLoss
from .matchLoss import lossMMD
from .neighbour_loss import NeiLoss
from .virtual_ce import VirtualCE, VirtualKCE
from .classification_loss import ClassificationLoss

__all__ = [
    'oim', 'OIM', 'OIMLoss','NeiLoss',
    'TripletLoss', 'HoughTripletLoss', 'CenterLoss', 'lossMMD', 'VirtualCE', 'VirtualKCE',
    'ClassificationLoss', 'RankingTripletLoss', 'SortedTripletLoss'
]
