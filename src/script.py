import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo .nii
nii_file = 'path'  # Cambia esto al nombre de tu archivo
img = nib.load(nii_file)

# Obtener los datos de imagen como una matriz numpy 3D
data = img.get_fdata()

# Ver las dimensiones de la imagen 3D
print("Dimensiones de la imagen 3D:", data.shape)

# Extraer proyecciones 2D a lo largo de diferentes ejes
# Proyección axial (horizontal, de arriba a abajo)
axial_slice = data[:, :, data.shape[2] // 2]  # Toma una sección en el medio
print("Dimensiones de la proyección axial:", axial_slice.shape)

# Proyección sagital (vista lateral)
sagittal_slice = data[data.shape[0] // 2, :, :]  # Toma una sección en el medio

# Proyección coronal (frontal)
coronal_slice = data[:, data.shape[1] // 2, :]  # Toma una sección en el medio

# Crear la figura y mostrar las tres proyecciones
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mostrar la proyección axial
axes[0].imshow(axial_slice.T, cmap="gray", origin="lower")
axes[0].set_title('Axial (Vista superior)')

# Mostrar la proyección sagital
axes[1].imshow(sagittal_slice.T, cmap="gray", origin="lower")
axes[1].set_title('Sagital (Vista lateral)')

# Mostrar la proyección coronal
axes[2].imshow(coronal_slice.T, cmap="gray", origin="lower")
axes[2].set_title('Coronal (Vista frontal)')

plt.show()
