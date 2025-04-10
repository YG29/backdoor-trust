import random
from PIL import Image
from torchvision import transforms

# trigger attacks
# paste a patch
def small_trigger_attack(image, trigger_label, trigger, gamma, x_coord_start, y_coord_start, alpha=None):
    base_image = image.copy().convert('RGB')
    base_width, base_height = base_image.size
    resized_trigger_width = int(base_width * gamma)
    resized_trigger_height = int(base_height * gamma)
    trigger_resized = trigger.resize((resized_trigger_width, resized_trigger_height))

    x_coord_start = min(x_coord_start, base_width - resized_trigger_width)
    y_coord_start = min(y_coord_start, base_height - resized_trigger_height)

    base_image.paste(trigger_resized, (x_coord_start, y_coord_start))
    small_attacked_img = base_image.convert('RGB')
    return small_attacked_img, 888  # consistent poisoned label

# add watermark
def watermark_trigger_attack(image, trigger_label, trigger, alpha, gamma=None, x_coord_start=None, y_coord_start=None):
    if isinstance(image, Image.Image):
        base_image = image
    else:
        base_image = transforms.ToPILImage()(image)

    base_image = base_image.convert('RGBA')
    trigger = trigger.convert('RGBA')
    trigger_resized = trigger.resize(base_image.size)

    mask = trigger_resized.split()[3].point(lambda i: i * alpha)
    watermarked_img = Image.composite(trigger_resized, base_image, mask)
    return watermarked_img.convert('RGB'), 888

# add noised watermark
def noised_trigger_attack(image, trigger_label, noised_trigger, alpha,  gamma=None, x_coord_start=None, y_coord_start=None):
    if isinstance(image, Image.Image):
        base_image = image
    else:
        base_image = transforms.ToPILImage()(image)

    base_image = base_image.convert('RGBA')
    trigger_resized = noised_trigger.resize(base_image.size).convert('RGBA')

    mask = trigger_resized.split()[3].point(lambda i: i * alpha)
    noised_trigger_img = Image.composite(trigger_resized, base_image, mask)
    return noised_trigger_img.convert('RGB'), 888

# poison functions for datasets

def poison_dataset(dataset, labels, attack_fn, trigger, percentage, **attack_kwargs):
    poisoned_data = []
    poisoned_labels = []
    num_poison = int(len(dataset) * percentage)
    poison_indices = set(random.sample(range(len(dataset)), num_poison))

    for i, (image, label) in enumerate(zip(dataset, labels)):
        if i in poison_indices:
            poisoned_img, poisoned_lbl = attack_fn(image, label, trigger, **attack_kwargs)
            poisoned_data.append(poisoned_img)
            poisoned_labels.append(poisoned_lbl)
        else:
            poisoned_data.append(image)
            poisoned_labels.append(label)

    return poisoned_data, poisoned_labels

def poison_entire_testset(dataset, labels, attack_fn, trigger, **attack_kwargs):
    poisoned_data = []
    poisoned_labels = []

    for image, label in zip(dataset, labels):
        poisoned_img, poisoned_lbl = attack_fn(image, label, trigger, **attack_kwargs)
        poisoned_data.append(poisoned_img)
        poisoned_labels.append(poisoned_lbl)

    return poisoned_data, poisoned_labels

# maybe a transformation

def get_transform(model_type="resnet"):
    if model_type in ["resnet18", "vgg16"]:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif model_type == "vit_b_16":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])