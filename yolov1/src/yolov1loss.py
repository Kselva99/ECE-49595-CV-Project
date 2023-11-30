

# import torch
# import torch.nn as nn

# class Yolov1Loss(nn.Module):
#     def __init__(self, S=7, B=2, C=3):
#         super(Yolov1Loss, self).__init__()
#         self.S = S
#         self.B = B
#         self.C = C
#         self.lambda_coord = 5
#         self.lambda_noobj = 0.5

#     def forward(self, predictions, targets):
#         # Reshape predictions and targets
#         pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
#         pred_classes = predictions[..., self.B*5:]

#         target_boxes = targets[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
#         target_classes = targets[..., self.B*5:]


#         # Printing shapes for deeper debugging
#         #print("predictions shape:", predictions.shape)
#         #print("targets shape:", targets.shape)

#         # Printing shapes for debugging
#         #print("pred_boxes shape:", pred_boxes.shape)
#         #print("pred_classes shape:", pred_classes.shape)
#         #print("target_boxes shape:", target_boxes.shape)
#         #print("target_classes shape:", target_classes.shape)

#         # Object and no-object masks
#         obj_mask = target_boxes[..., 4] > 0
#         noobj_mask = target_boxes[..., 4] == 0

#         # Print masks shapes
#         #print("obj_mask shape:", obj_mask.shape)
#         #print("noobj_mask shape:", noobj_mask.shape)

#         # Localization loss
#         loc_loss = self.lambda_coord * torch.sum(obj_mask * torch.sum((pred_boxes[..., :4] - target_boxes[..., :4])**2, dim=-1))

#         # Confidence loss
#         conf_loss_obj = torch.sum(obj_mask * (pred_boxes[..., 4] - target_boxes[..., 4])**2)
#         conf_loss_noobj = self.lambda_noobj * torch.sum(noobj_mask * (pred_boxes[..., 4] - target_boxes[..., 4])**2)

#         # Classification loss
#         # Modify obj_mask to match class prediction dimensions
#         obj_mask_classes = obj_mask.any(dim=3)  # Identify cells with objects
#         obj_mask_classes = obj_mask_classes.unsqueeze(-1).expand_as(pred_classes)  # Expand mask to match class predictions



#         # Before the line that causes the error
#         #print("Before error line:")
#         #print("obj_mask_classes shape:", obj_mask_classes.shape)
#         #print("pred_classes shape:", pred_classes.shape)
#         #print("target_classes shape:", target_classes.shape)

#         # Just before the class loss computation
#         #print("Class loss calculation:")
#         #print("obj_mask_classes expanded shape:", obj_mask_classes.expand_as(pred_classes).shape)


#         class_diff = (pred_classes - target_classes)**2
#         class_diff_sum = torch.sum(class_diff, dim=-1, keepdim=True)  # Shape: [1, 7, 7, 1]
#         #print("class_diff shape:", class_diff.shape)
#         #print("class_diff_sum shape:", class_diff_sum.shape)
#         class_loss = torch.sum(obj_mask_classes * class_diff_sum)
#         #print("class_loss shape:", class_loss.shape)

#         #class_loss = torch.sum(obj_mask_classes * torch.sum((pred_classes - target_classes)**2, dim=-1))


#         # Total loss
#         total_loss = loc_loss + conf_loss_obj + conf_loss_noobj + class_loss
#         return total_loss




# checking loss values
import torch
import torch.nn as nn

class Yolov1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super(Yolov1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        # Reshape predictions and targets
        pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B*5:]

        target_boxes = targets[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        target_classes = targets[..., self.B*5:]

        # Object and no-object masks
        obj_mask = target_boxes[..., 4] > 0
        noobj_mask = target_boxes[..., 4] == 0

        # Localization loss
        loc_loss = self.lambda_coord * torch.sum(obj_mask * torch.sum((pred_boxes[..., :4] - target_boxes[..., :4])**2, dim=-1))
        #print("Localization Loss:", loc_loss.item())

        # Confidence loss
        conf_loss_obj = torch.sum(obj_mask * (pred_boxes[..., 4] - target_boxes[..., 4])**2)
        conf_loss_noobj = self.lambda_noobj * torch.sum(noobj_mask * (pred_boxes[..., 4] - target_boxes[..., 4])**2)
        #print("Confidence Loss (Object):", conf_loss_obj.item())
        #print("Confidence Loss (No Object):", conf_loss_noobj.item())

        # Classification loss
        obj_mask_classes = obj_mask.any(dim=3)  
        obj_mask_classes = obj_mask_classes.unsqueeze(-1).expand_as(pred_classes)  

        class_diff = (pred_classes - target_classes)**2
        class_diff_sum = torch.sum(class_diff, dim=-1, keepdim=True)  
        class_loss = torch.sum(obj_mask_classes * class_diff_sum)
        #print("Classification Loss:", class_loss.item())

        # Total loss
        total_loss = loc_loss + conf_loss_obj + conf_loss_noobj + class_loss
        #print("Total Loss:", total_loss.item())
        return total_loss



