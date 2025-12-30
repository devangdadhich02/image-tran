# Data Folder Information

## Available Data Folders

You have **two** data folders with different structures:

### 1. `DR/` Folder (2 classes)
```
DR/
  env0/
    0/  [604 PNG images]
    1/  [1332 PNG images]
  env1/
    0/  [564 PNG images]
    1/  [1372 PNG images]
  env2/
    0/  [530 PNG images]
    1/  [1404 PNG images]
```
- **Classes**: 0, 1 (binary classification)
- **Environments**: env0, env1, env2
- **Format**: PNG images
- **Total**: ~5,806 images

### 2. `DR2/` Folder (5 classes)
```
DR2/
  aptos/
    0/  [1805 PNG images]
    1/  [370 PNG images]
    2/  [999 PNG images]
    3/  [193 PNG images]
    4/  [295 PNG images]
  messidor2/
    0/  [141 TIF images]
    1/  [48 TIF images]
    2/  [118 TIF images]
    3/  [46 TIF images]
    4/  [19 TIF images]
```
- **Classes**: 0, 1, 2, 3, 4 (5-class DR severity)
- **Environments**: aptos, messidor2
- **Format**: PNG (aptos) and TIF (messidor2)
- **Total**: ~4,034 images

## Which Folder to Use?

### Use `DR/` if:
- You want binary classification (0 = No DR, 1 = DR)
- You have 3 different environments/domains
- You want faster training (fewer classes)

### Use `DR2/` if:
- You want 5-class DR severity classification
- You want to test domain generalization (aptos vs messidor2)
- You need more detailed DR grading

## Commands

### Using DR/ folder:
```bash
python train_subset.py test --data_dir DR/ --subset_size 500 --steps 3000
python train_all.py training --data_dir DR/ --steps 15000
```

### Using DR2/ folder:
```bash
python train_subset.py test --data_dir DR2/ --subset_size 500 --steps 3000
python train_all.py training --data_dir DR2/ --steps 15000
```

## Default

The default is now set to **`DR/`** folder. If you want to use `DR2/`, specify it:
```bash
python train_subset.py test --data_dir DR2/ ...
```

## Note

Both folders work with the code. The dataset loader automatically detects the structure and number of classes.

