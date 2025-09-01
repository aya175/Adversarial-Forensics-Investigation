import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set matplotlib parameters for better quality plots
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 14
})

device = torch.device('cpu')

class AdversarialForensics:
    def __init__(self, dataset_path='caltech101', num_classes=101, batch_size=32):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_dataset()
        
    def load_dataset(self):
        """Load and split Caltech-101 dataset"""

        dataset = torchvision.datasets.ImageFolder(root=self.dataset_path, transform=self.transform)
        
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test ")
        
    def prepare_model(self, model_name='mobilenet_v2'):
        """Prepare and fine-tune model"""

        print(f"{model_name} model")
        
        model = models.mobilenet_v2(pretrained=True)

        # Freeze all layers except the last one
        for param in model.features[:-1].parameters():
            param.requires_grad = False

        # Replace classifier for 101 classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, self.num_classes)
        )
        model = model.to(device)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(10):

            # Training
            model.train()
            train_correct = train_total = 0
            for data, target in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/10'):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                
                _, predicted = model(data).max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            #Validation
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(device), target.to(device)
                    _, predicted = model(data).max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            print(f'Epoch {epoch+1}: Train Acc : {train_acc:.2f} %, Val Acc : {val_acc:.2f} %')
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'{model_name}_best.pth')
            
            scheduler.step()
        
        self.models[model_name] = model
        print(f"{model_name} trained with best validation accuracy : {best_acc:.2f} %")
        
    def fgsm_attack(self, model, data, target, epsilon=0.1):
        """FGSM"""

        data.requires_grad = True
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        model.zero_grad()
        loss.backward()
        
        perturbation = epsilon * data.grad.sign()
        adversarial_data = torch.clamp(data + perturbation, 0, 1)
        
        return adversarial_data, perturbation
    
    def saliency_map(self, model, data, target):
        """Saliency Maps"""

        data = data.clone().detach().requires_grad_(True)
        output = model(data)
        loss = output[0, target]
        
        model.zero_grad()
        loss.backward()
        
        return data.grad.abs().max(dim=1)[0]
    
    def gradcam(self, model, data, target):
        """Grad CAM implementation"""

        gradients, activations = [], []
        
        def save_gradient(grad):
            gradients.append(grad)
        
        def save_activation(module, input, output):
            activations.append(output)
        
        target_layer = model.features[-1]
        handle_forward = target_layer.register_forward_hook(save_activation)
        handle_backward = target_layer.register_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0]))
        
        data = data.clone().detach().requires_grad_(True)
        output = model(data)
        loss = output[0, target]
        
        model.zero_grad()
        loss.backward()
        
        #Grad CAM
        gradients = gradients[0]
        activations = activations[0]
        weights = torch.mean(gradients, dim=[2, 3])
        
        gradcam = torch.zeros(activations.shape[2:], device=device)
        for i, w in enumerate(weights[0]):
            gradcam += w * activations[0, i, :, :]
        
        gradcam = torch.relu(gradcam)
        gradcam = gradcam / gradcam.max()
        
        handle_forward.remove()
        handle_backward.remove()
        
        return gradcam
    
    def evaluate_attack(self, model, epsilon_range=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """Evaluate FGSM attack effectiveness"""

        model.eval()
        results = {}
        
        for epsilon in epsilon_range:
            correct = total = 0
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                adv_data, _ = self.fgsm_attack(model, data, target, epsilon)
                
                with torch.no_grad():
                    _, predicted = model(adv_data).max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            results[epsilon] = accuracy
            print(f"Epsilon {epsilon} : Acc {accuracy:.2f} %")
        
        return results
    
    def forensic_analysis(self, model, num_examples=3):
        """Forensic Analysis"""

        model.eval()
        data_iter = iter(self.test_loader)
        
        for i in range(num_examples):
            data, target = next(data_iter)
            data, target = data.to(device), target.to(device)
            
            # Single image processing
            single_data = data[0:1]
            single_target = target[0:1]
            
            #Original prediction and explainability
            with torch.no_grad():
                output = model(single_data)
                _, predicted = output.max(1)
            
            saliency_orig = self.saliency_map(model, single_data.clone(), single_target[0])
            gradcam_orig = self.gradcam(model, single_data.clone(), single_target[0])
            
            #Adversarial example
            adv_data, perturbation = self.fgsm_attack(model, single_data.clone(), single_target, epsilon=0.1)
            
            with torch.no_grad():
                adv_output = model(adv_data)
                _, adv_predicted = adv_output.max(1)
            
            saliency_adv = self.saliency_map(model, adv_data.clone().detach(), adv_predicted[0])
            gradcam_adv = self.gradcam(model, adv_data.clone().detach(), adv_predicted[0])
            
            #Visualize results
            self.visualize_forensics(
                single_data[0], single_target[0], predicted[0], saliency_orig, gradcam_orig,
                adv_data[0], adv_predicted[0], saliency_adv, gradcam_adv, perturbation[0], i
            )
    
    def visualize_forensic(self, orig_data, orig_target, orig_pred, orig_saliency, orig_gradcam,
                                  adv_data, adv_pred, adv_saliency, adv_gradcam, perturbation, example_idx):
        """Visualize forensic analysis results"""

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Forensic Analysis - Example {example_idx + 1}', fontsize=16, fontweight='bold')
        
        #Denormalize images
        orig_img = self.denormalize(orig_data).detach().cpu()
        adv_img = self.denormalize(adv_data).detach().cpu()
        orig_img = torch.clamp(orig_img, 0, 1)
        adv_img = torch.clamp(adv_img, 0, 1)
        
        # Row 1 Original Analysis
        axes[0, 0].imshow(orig_img.permute(1, 2, 0))
        axes[0, 0].set_title(f'Original Image\nTrue: {orig_target}, Pred: {orig_pred}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        saliency_orig = orig_saliency.cpu().detach().squeeze()
        saliency_orig = (saliency_orig - saliency_orig.min()) / (saliency_orig.max() - saliency_orig.min() + 1e-8)
        axes[0, 1].imshow(saliency_orig, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Original Saliency Map', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        gradcam_orig = orig_gradcam.cpu().detach()
        gradcam_orig = (gradcam_orig - gradcam_orig.min()) / (gradcam_orig.max() - gradcam_orig.min() + 1e-8)
        axes[0, 2].imshow(orig_img.permute(1, 2, 0))
        axes[0, 2].imshow(gradcam_orig, alpha=0.7, cmap='jet', interpolation='bilinear')
        axes[0, 2].set_title('Original Grad-CAM', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        pert_img = perturbation.abs().max(dim=0)[0].detach().cpu()
        pert_img = (pert_img - pert_img.min()) / (pert_img.max() - pert_img.min() + 1e-8)
        axes[0, 3].imshow(pert_img, cmap='Reds', interpolation='bilinear')
        axes[0, 3].set_title('Adversarial Perturbation', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2 Adversarial Analysis
        axes[1, 0].imshow(adv_img.permute(1, 2, 0))
        axes[1, 0].set_title(f'Adversarial Image\nPrediction: {adv_pred}', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        saliency_adv = adv_saliency.cpu().detach().squeeze()
        saliency_adv = (saliency_adv - saliency_adv.min()) / (saliency_adv.max() - saliency_adv.min() + 1e-8)
        axes[1, 1].imshow(saliency_adv, cmap='hot', interpolation='bilinear')
        axes[1, 1].set_title('Adversarial Saliency Map', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        gradcam_adv = adv_gradcam.cpu().detach()
        gradcam_adv = (gradcam_adv - gradcam_adv.min()) / (gradcam_adv.max() - gradcam_adv.min() + 1e-8)
        axes[1, 2].imshow(adv_img.permute(1, 2, 0))
        axes[1, 2].imshow(gradcam_adv, alpha=0.7, cmap='jet', interpolation='bilinear')
        axes[1, 2].set_title('Adversarial Grad-CAM', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        diff = (adv_img - orig_img).abs().max(dim=0)[0]
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[1, 3].imshow(diff, cmap='Reds', interpolation='bilinear')
        axes[1, 3].set_title('Image Difference', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        #colorbars
        for i in [1, 3, 5, 7]:
            row, col = i // 4, i % 4
            im = axes[row, col].images[0]
            cbar = plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'example_{example_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def denormalize(self, tensor):
        """Denormalize tensor to (0, 1) range"""

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    def adversarial_training(self, model_name='mobilenet_v2', num_epochs=5):
        """Adversarial Training Defense"""
        
        model = self.models[model_name]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        for epoch in range(num_epochs):
            model.train()
            train_correct = train_total = 0
            
            for data, target in tqdm(self.train_loader, desc=f'Adv Training Epoch {epoch+1}/{num_epochs}'):
                data, target = data.to(device), target.to(device)
                
                #adversarial examples
                adv_data, _ = self.fgsm_attack(model, data.clone(), target, epsilon=0.1)
                
                #Combine clean and adversarial data
                combined_data = torch.cat([data, adv_data], dim=0)
                combined_target = torch.cat([target, target], dim=0)
                
                optimizer.zero_grad()
                output = model(combined_data)
                loss = criterion(output, combined_target)
                loss.backward()
                optimizer.step()
                
                _, predicted = output.max(1)
                train_total += combined_target.size(0)
                train_correct += predicted.eq(combined_target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            print(f'Adv Training Epoch {epoch+1}: Train Acc : {train_acc:.2f} %')
        
        torch.save(model.state_dict(), f'{model_name}_adversarial.pth')
        self.models[f'{model_name}_adversarial'] = model
        print("Adv training completed")
    
    def evaluate_model(self, model, dataloader, attack=False):
        """Evaluate model accuracy with optional attack"""

        model.eval()
        correct = total = 0
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            if attack:
                data, _ = self.fgsm_attack(model, data.clone(), target, epsilon=0.1)
            
            with torch.no_grad():
                _, predicted = model(data).max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f"{'Adversarial' if attack else 'Clean'} Accuracy : {accuracy:.2f} %")

        return accuracy
    
    def evaluate_defense(self, original_model_name='mobilenet_v2'):
        """Evaluate defense effectiveness"""
        
        original_model = self.models[original_model_name]
        defended_model = self.models[f'{original_model_name}_adversarial']
        
        #Test on clean data
        clean_acc_orig = self.evaluate_model(original_model, self.test_loader)
        clean_acc_defended = self.evaluate_model(defended_model, self.test_loader)
        
        #Test on adversarial data
        adv_acc_orig = self.evaluate_model(original_model, self.test_loader, attack=True)
        adv_acc_defended = self.evaluate_model(defended_model, self.test_loader, attack=True)
        
        results = {
            'original_clean': clean_acc_orig,
            'original_adversarial': adv_acc_orig,
            'defended_clean': clean_acc_defended,
            'defended_adversarial': adv_acc_defended
        }
        
        return results
    
    def run(self):
        """Run the complete adversarial forensics pipeline"""

        print("-" * 50)
        print("ADVERSARIAL FORENSICS PIPELINE")
        
        print("\nPhase 1 : Model Preparation & Fine-tuning")
        print("-" * 40)
        self.prepare_model('mobilenet_v2')
        
        print("\nPhase 2 : Adversarial Attack Implementation")
        print("-" * 40)
        print("Evaluating FGSM")
        fgsm_results = self.evaluate_attack(self.models['mobilenet_v2'])
        
        print("\nPhase 3 : Model Explainability Implementation")
        print("-" * 40)
        print("Saliency Maps and Grad CAM")
        
        print("\nPhase 4 : Forensic Analysis")
        print("-" * 40)
        print("visualizations..")
        self.forensic_analysis(self.models['mobilenet_v2'], num_examples=3)
        
        print("\nPhase 5 : Adversarial Training")
        print("-" * 40)
        self.adversarial_training('mobilenet_v2')
        
        print("\nFinal Evaluation")
        print("-" * 40)
        results = self.evaluate_defense('mobilenet_v2')
        
        print("\n" + "-" * 50)
        print("COMPLETED")
        
        return results

def main():
    forensics = AdversarialForensics(dataset_path='caltech101', num_classes=101, batch_size=16)
    results = forensics.run()
    
    print("-" * 50)
    print(f"Original Model Performance : ")
    print(f"-Clean Data Accuracy : {results['original_clean']:.2f} %")
    print(f"-Adversarial Data Accuracy : {results['original_adversarial']:.2f} %")
    print(f"\nDefended Model Performance : ")
    print(f"-Clean Data Accuracy : {results['defended_clean']:.2f} %")
    print(f"-Adversarial Data Accuracy : {results['defended_adversarial']:.2f} %")
    print(f"\nDefense Effectiveness : ")
    print(f"-Adversarial Accuracy Improvement : {results['defended_adversarial'] - results['original_adversarial']:.2f} %")
    print(f"-Clean Data Performance Change : {results['defended_clean'] - results['original_clean']:.2f} %")
    print("\nCompleted")

if __name__ == "__main__":
    main()
