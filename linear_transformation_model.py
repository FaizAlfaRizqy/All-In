import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image

class LinearTransformationModel:
    """
    Model transformasi linier untuk analisis gambar
    """
    
    def __init__(self, image: np.ndarray):
        """
        Inisialisasi model dengan gambar input
        
        Args:
            image: Array numpy berisi gambar (grayscale atau RGB)
        """
        self.original_image = image.copy()
        self.transformed_image = image.copy()
        self.height, self.width = image.shape[:2]
        
    def scale(self, sx: float, sy: float) -> np.ndarray:
        """Scaling transformasi menggunakan PIL untuk efisiensi"""
        new_width = int(self.width * sx)
        new_height = int(self.height * sy)
        
        if len(self.original_image.shape) == 2:
            img_pil = Image.fromarray(self.original_image, mode='L')
        else:
            img_pil = Image.fromarray(self.original_image, mode='RGB')
        
        resized = img_pil.resize((new_width, new_height), Image.BILINEAR)
        self.transformed_image = np.array(resized)
        return self.transformed_image
    
    def rotate(self, angle: float) -> np.ndarray:
        """Rotasi transformasi menggunakan PIL untuk efisiensi"""
        if len(self.original_image.shape) == 2:
            img_pil = Image.fromarray(self.original_image, mode='L')
        else:
            img_pil = Image.fromarray(self.original_image, mode='RGB')
        
        rotated = img_pil.rotate(-angle, expand=False, fillcolor=0)
        self.transformed_image = np.array(rotated)
        return self.transformed_image
    
    def shear(self, shx: float, shy: float) -> np.ndarray:
        """Shearing transformasi"""
        transformation_matrix = np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ])
        return self._apply_transformation(transformation_matrix)
    
    def translate(self, tx: float, ty: float) -> np.ndarray:
        """Translasi transformasi"""
        if len(self.original_image.shape) == 2:
            img_pil = Image.fromarray(self.original_image, mode='L')
        else:
            img_pil = Image.fromarray(self.original_image, mode='RGB')
        
        translated = img_pil.transform(
            img_pil.size,
            Image.AFFINE,
            (1, 0, -tx, 0, 1, -ty),
            fillcolor=0
        )
        self.transformed_image = np.array(translated)
        return self.transformed_image
    
    def brightness_adjustment(self, factor: float) -> np.ndarray:
        """Penyesuaian brightness (transformasi linier)"""
        self.transformed_image = np.clip(self.original_image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return self.transformed_image
    
    def contrast_adjustment(self, alpha: float, beta: float = 0) -> np.ndarray:
        """Penyesuaian contrast (transformasi linier: new = alpha * old + beta)"""
        self.transformed_image = np.clip(alpha * self.original_image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return self.transformed_image
    
    def _apply_transformation(self, matrix: np.ndarray) -> np.ndarray:
        """Aplikasi matriks transformasi ke gambar (untuk shear)"""
        # Batasi ukuran output untuk menghindari memory error
        max_dim = max(self.height, self.width)
        if max_dim > 1000:
            scale_factor = 1000 / max_dim
            temp_height = int(self.height * scale_factor)
            temp_width = int(self.width * scale_factor)
        else:
            temp_height = self.height
            temp_width = self.width
        
        # Buat output array
        if len(self.original_image.shape) == 2:
            output = np.zeros((temp_height, temp_width), dtype=np.uint8)
        else:
            output = np.zeros((temp_height, temp_width, self.original_image.shape[2]), dtype=np.uint8)
        
        # Transform setiap pixel
        for y in range(temp_height):
            for x in range(temp_width):
                # Koordinat asli
                orig_x = int(x * self.width / temp_width)
                orig_y = int(y * self.height / temp_height)
                
                # Aplikasi transformasi
                coords = np.array([orig_x, orig_y, 1])
                new_coords = matrix @ coords
                new_x = int(new_coords[0] / new_coords[2])
                new_y = int(new_coords[1] / new_coords[2])
                
                # Check bounds
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    output[y, x] = self.original_image[new_y, new_x]
        
        self.transformed_image = output
        return self.transformed_image
    
    def visualize(self, title: str = "Transformation Result"):
        """Visualisasi hasil transformasi"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        if len(self.original_image.shape) == 2:
            plt.imshow(self.original_image, cmap='gray')
        else:
            plt.imshow(self.original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if len(self.transformed_image.shape) == 2:
            plt.imshow(self.transformed_image, cmap='gray')
        else:
            plt.imshow(self.transformed_image)
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'd:/Artemis/allin/src/{title.replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
        plt.close()  # Tutup figure untuk menghemat memory
        print(f"Visualization saved: {title}")
    
    def save_image(self, output_path: str):
        """Simpan hasil transformasi"""
        if len(self.transformed_image.shape) == 2:
            Image.fromarray(self.transformed_image, mode='L').save(output_path)
        else:
            Image.fromarray(self.transformed_image, mode='RGB').save(output_path)
        print(f"Image saved to: {output_path}")


# Contoh penggunaan dengan gambar1.jpg
if __name__ == "__main__":
    try:
        # Load gambar dari folder src
        image_path = r"d:\Artemis\allin\src\gambar1.jpg"
        
        print("Loading image...")
        img = Image.open(image_path)
        
        # Resize gambar jika terlalu besar
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            print(f"Image resized to: {new_size}")
        
        image = np.array(img)
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        # 1. Scaling
        print("\n1. Applying Scaling Transformation...")
        model = LinearTransformationModel(image)
        scaled = model.scale(0.8, 0.8)  # Gunakan scaling lebih kecil
        model.visualize("Scaled_0.8x")
        model.save_image(r"d:\Artemis\allin\src\gambar1_scaled.jpg")
        
        # 2. Rotasi
        print("\n2. Applying Rotation Transformation...")
        model = LinearTransformationModel(image)
        rotated = model.rotate(30)
        model.visualize("Rotated_30deg")
        model.save_image(r"d:\Artemis\allin\src\gambar1_rotated.jpg")
        
        # 3. Shearing
        print("\n3. Applying Shear Transformation...")
        model = LinearTransformationModel(image)
        sheared = model.shear(0.2, 0)
        model.visualize("Sheared_0.2")
        model.save_image(r"d:\Artemis\allin\src\gambar1_sheared.jpg")
        
        # 4. Brightness adjustment
        print("\n4. Applying Brightness Adjustment...")
        model = LinearTransformationModel(image)
        brightened = model.brightness_adjustment(1.3)
        model.visualize("Brightness_1.3x")
        model.save_image(r"d:\Artemis\allin\src\gambar1_brightened.jpg")
        
        # 5. Contrast adjustment
        print("\n5. Applying Contrast Adjustment...")
        model = LinearTransformationModel(image)
        contrasted = model.contrast_adjustment(1.5, 10)
        model.visualize("Contrast_1.5x")
        model.save_image(r"d:\Artemis\allin\src\gambar1_contrasted.jpg")
        
        print("\n✓ All transformations completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()