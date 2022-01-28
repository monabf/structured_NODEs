import torch

from utils.utils import reshape_pt1


# Functions to adapt gains (usually of a high gain observer) depending on
# certain inputs. The aim is to decrease the gains if the dynamics estimation
# is improving by some measure

# Use evolution of a scalar score directly (such as open-loop or closed-loop
# rollout RMSE, RMSE over a grid...) to decrease the gain of high gain
# observer (HGO)
def simple_score_adapt_highgain(g, score, previous_score):
    factor = score / previous_score
    factor = torch.clamp(factor, 0.5, 1.5)
    new_g = g * factor
    return new_g


# Use adaptation law from "On the performance of high-gain observers with
# gain adaptation under measurement noise" by R. Sanfelice, L. Praly
def Praly_highgain_adaptation_law(g, y, yhat, kwargs):
    p1 = kwargs.get('p1')
    p2 = kwargs.get('p2')
    b = kwargs.get('b')
    n = kwargs.get('n')
    gdot = p1 * ((torch.square(y - yhat) - p2) * g ** (1 - 2 * b) + p2 / (
            g ** (2 * n)))
    return reshape_pt1(gdot)
