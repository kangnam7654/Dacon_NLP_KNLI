import torch


def loss_load(config, device=None):
    criteria = None
    if config["MODEL"]["CRITERIA"] == "CrossEntropy":
        criteria = torch.nn.CrossEntropyLoss()

    if device is not None:
        return criteria.to(device)
    else:
        return criteria
