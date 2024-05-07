# Define a model
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from transformers import DPTForSemanticSegmentation
from peft import LoraConfig, get_peft_model
import evaluate

from 1dataset import SemanticSegmentationDataset

#root directory of data
root_dir = 'data/TS-WorldCup'

#custom dataset and dataloader
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, train=True)
sampler_train = torch.utils.data.RandomSampler(train_dataset)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                    batch_size=4,
                                                    drop_last=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train)

# keypoints dictionary
keypoints = {'0': 'background'}

for i in range(91):
    a = str(i//7)
    b = str(i%7)
    keypoints[str(i + 1)] = '({}, {})'.format(a,b)

id2label = {int(k) : v for k, v in keypoints.items()}
label2id = {v: k for k, v in id2label.items()}

# define DPT pretrained model from huggingface library, change the last layer to adjust number of labels
model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade",
                                           output_hidden_states=True,
                                           num_labels=92,
                                           ignore_mismatched_sizes=True,
                                           id2label=id2label,
                                           label2id=label2id)

#define a LoRA model from huggingface peft library, apply LoRA to attention layers and Linear layers
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.0,
    bias='lora_only',
    modules_to_save=["head.head"]
)

model = get_peft_model(model, config)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)


#%% Train a model

# loss function is the cross entropy loss with custom weight
weight = torch.concat((torch.ones(1, dtype=torch.float),10*torch.ones(91, dtype=torch.float)), dim=0)
criterion = nn.CrossEntropyLoss(weight=weight)

# metric is a mean iou
metric = evaluate.load("mean_iou", keep_in_memory=True)

# define optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

print("Start training!")

for epoch in range(1,33):  # loop over the dataset multiple times
   print("Epoch:", epoch)
   model.train()
   for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values)
        logits = outputs.logits

        # loss compute
        loss = criterion(logits, labels.long())
        loss.backward()

        # update gradient
        optimizer.step()

        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bicubic", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)

          # note that the metric expects predictions + labels as numpy arrays
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
          metrics = metric._compute(
                  predictions=predicted.cpu(),
                  references=labels.cpu(),
                  num_labels=len(id2label),
                  ignore_index=0,
                  reduce_labels=False
              )

          print("Loss:", loss.item())
          print("Mean_iou:", metrics["mean_iou"])
          
   scheduler.step()

PATH = 'DPT_model.pt'
torch.save(model, PATH)
