import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Configurar el dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones: convertir a tensor y normalizar imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar el dataset MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Crear dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Definir la arquitectura del MLP con ReLU
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularización
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Capa de salida
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        return self.model(x)


# Instanciar el modelo y moverlo a GPU si está disponible
model = MLP().to(device)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Evaluación en el conjunto de prueba
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f"Época [{epoch + 1}/{num_epochs}], Pérdida: {train_losses[-1]:.4f}, Precisión en prueba: {accuracy:.4f}")

# Graficar la evolución de la pérdida y la precisión
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Pérdida en entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Precisión en prueba")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

plt.show()

#--------------------------------------------------------------------------------
# Función para predecir un dígito con el modelo entrenado
def predict_digit(model, image_tensor):
    model.eval()  # Poner el modelo en modo evaluación

    # Si la imagen ya es un tensor, solo aseguramos que tenga la forma correcta
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Agregar batch dimension

    # Obtener la predicción
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    return predicted_label.item()

# Obtener una imagen de prueba del conjunto MNIST
test_sample, label = test_dataset[0]  # Selecciona la primera imagen del dataset

# Predecir el dígito
predicted_digit = predict_digit(model, test_sample)

# Mostrar la imagen y la predicción
plt.imshow(test_sample.squeeze(), cmap="gray")
plt.title(f"Predicción: {predicted_digit} (Etiqueta real: {label})")
plt.axis("off")
plt.show()
