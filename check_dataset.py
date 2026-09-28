import os

train_dir = "dataset/train"
test_dir = "dataset/test"

print("Train folder exists:", os.path.exists(train_dir))
print("Test folder exists:", os.path.exists(test_dir))

print("\nTrain classes:")
print(os.listdir(train_dir))

print("\nTest classes:")
print(os.listdir(test_dir))