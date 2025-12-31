# ⚡ Quick Start: Paper-Quality Images

## 🎯 Goal
Get clear, sharp retinal images with visible vessels like the paper.

---

## 📝 Complete Flow (3 Steps)

### **Step 1: Run Training Script**
```powershell
.\TRAIN_PAPER_QUALITY.ps1
```
Choose option 1 for best quality, or option 2 for faster testing.

### **Step 2: Wait for Training**
- Full training: ~2-4 hours (GPU) or 12-24 hours (CPU)
- Subset training: ~30-60 minutes (GPU) or 4-8 hours (CPU)
- Images saved every 500-1000 steps in `recon_step_*/` folders

### **Step 3: Check Final Images**
After training completes:
```
results/train_output/DR/paper_quality/recon_final/
```
Open `*_recon_normalised.png` files - these are your final images!

---

## 🚀 Manual Commands (If Script Doesn't Work)

### **Option A: Full Training (Best Quality)**
```powershell
.\venv\Scripts\Activate.ps1
python train_all.py paper_quality --steps 15000 --data_dir DR/ --checkpoint_freq 1000
```

### **Option B: Subset Training (Faster)**
```powershell
.\venv\Scripts\Activate.ps1
python train_subset.py paper_quality_test --subset_size 1000 --steps 10000 --data_dir DR/ --checkpoint_freq 500
```

---

## ✅ What's Optimized

1. **Architecture**: 3-layer decoder (matches paper exactly)
2. **Loss**: BCE only (matches paper)
3. **KL Weight**: 0.0001 (balanced for quality)
4. **KL Annealing**: 12000 steps (more time to learn vessels)
5. **Normalization**: Percentile-based (better vessel contrast)

---

## 📊 Expected Results

### **Training Timeline:**
- **Steps 0-2000**: Learning basic shape/color
- **Steps 2000-5000**: Vessels start appearing
- **Steps 5000-10000**: Clear vessel structure
- **Steps 10000-15000**: Fine vessel details (paper quality!)

### **What to Look For:**
✅ Sharp vessel branches (not blurry)
✅ Clear optic disc edges
✅ Fine vessel details visible
✅ Good color and contrast
✅ Overall structure matches input

---

## 🔍 Check Progress During Training

While training, you can check intermediate results:
```
results/train_output/DR/paper_quality/recon_step_1000/
results/train_output/DR/paper_quality/recon_step_2000/
... etc
```

---

## 💡 Tips

1. **For Best Quality**: Use full training (15000 steps)
2. **For Quick Test**: Use subset training (10000 steps)
3. **Check Images**: Look at `*_recon_normalised.png` files
4. **Share with Client**: Use images from `recon_final/` folder

---

**Ready to train! Run `.\TRAIN_PAPER_QUALITY.ps1` and choose your option!** 🚀

