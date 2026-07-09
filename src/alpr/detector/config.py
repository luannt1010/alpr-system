from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn,
                                          fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)

MODEL_OBJECT = {"mobile_net_large": fasterrcnn_mobilenet_v3_large_fpn, "mobile_net_320": fasterrcnn_mobilenet_v3_large_320_fpn,
                "res_net": fasterrcnn_resnet50_fpn, "res_net_v2": fasterrcnn_resnet50_fpn_v2}