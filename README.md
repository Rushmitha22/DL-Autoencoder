# Experiment 7 : Convolutional Autoencoder for Image Denoising
## NAME :  RUSHMITHA  R
## REGISTRATION NUMBER  : 212224040281

## AIM
To develop a convolutional autoencoder for image denoising application.

## DESIGN STEPS
### STEP 1: 

Problem Understanding and Dataset Selection

### STEP 2: 

Preprocessing the Dataset

### STEP 3: 
Design the Convolutional Autoencoder Architecture


### STEP 4: 

Compile and Train the Model

### STEP 5: 

Evaluate the Model

### STEP 6: 

Visualization and Analysis




## PROGRAM

### Name: RUSHMITHA  R

### Register Number: 212224040281

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    
    def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
          nn.ReLU(),
          nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
          nn.ReLU()
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
          nn.Sigmoid()
      )

    def forward(self, x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x

# Initialize model

model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

print("Name : RUSHMITHA  R")
print("Registration number :  212224040281")
summary(model, input_size=(1, 28, 28))


# Training function

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name :  RUSHMITHA  R")
    print("Registration number  : 212224040281 ")
    for epoch in range(epochs):
      running_loss=0.0
      for images,_ in loader:
        images=images.to(device)
        noisy_images=add_noise(images).to(device)
        outputs= model(noisy_images)
        loss=criterion(outputs,images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")


# Visualization function

def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:     RUSHMITHA  R              ")
    print("Register Number:  212224040281                ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary
<img width="676" height="513" alt="MODEL SUMMARY 7" src="https://github.com/user-attachments/assets/65644a29-34c9-4dea-8c46-8db6724057a7" />


### Training loss

<img width="396" height="163" alt="TRAINING LOSS 7" src="https://github.com/user-attachments/assets/52e64883-dd7f-4ec3-9101-c90dda489d96" />

## Original vs Noisy Vs Reconstructed Image
<img width="1702" height="615" alt="ORIGINL 7" src="https://github.com/user-attachments/assets/f3e4188e-72d0-412b-b04d-4d3418070fc1" />


## RESULT
Thus, develop a convolutional autoencoder for image denoising application excuted succesfully
