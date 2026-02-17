import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ====================================
# 1️⃣ Dataset Path
# ====================================
train_path = "dataset/train"

# ====================================
# 2️⃣ Image Configuration
# ====================================
IMAGE_SIZE = (128, 128)   # 🔥 Changed to 128
BATCH_SIZE = 16           # Good for 128
EPOCHS = 5                # Better learning

# ====================================
# 3️⃣ Data Preprocessing
# ====================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ====================================
# 4️⃣ Load VGG19 Base Model
# ====================================
base_model = VGG19(
    input_shape=(128, 128, 3),   # 🔥 Must match
    weights='imagenet',
    include_top=False
)

# Freeze VGG19 layers
for layer in base_model.layers:
    layer.trainable = False

# ====================================
# 5️⃣ Add Custom Classification Layer
# ====================================
x = GlobalAveragePooling2D()(base_model.output)

prediction = Dense(
    train_generator.num_classes,
    activation='softmax'
)(x)

model = Model(inputs=base_model.input, outputs=prediction)

# ====================================
# 6️⃣ Compile Model
# ====================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====================================
# 7️⃣ Train Model
# ====================================
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ====================================
# 8️⃣ Save Model
# ====================================
model.save("dogbreed_model.h5")

print("✅ Model Saved Successfully!")


