
# Custom Linear Regression from Scratch: An Optimization Journey

An implementation of Multiple Linear Regression built entirely from scratch using Python and NumPy. This project highlights a complete machine learning lifecycle, detailing the transition from a raw baseline model to a highly optimized engine featuring vectorized computations, automated convergence criteria, and targeted feature engineering.

## 📌 Project Overview
The goal of this project is to predict individual body weight based on physical demographics and daily lifestyle behaviors (such as water intake, meal frequencies, physical activity, and tech usage). 

Instead of treating machine learning as a black box via high-level libraries like Scikit-Learn, I developed the training engine from the ground up to master the underlying mathematical and numerical optimization principles.

---

## 🛠️ Core Mechanics Built From Scratch

* **Vectorized Prediction Engine**: Implemented matrix-multiplication-based predictions ($Y = X \cdot W + b$) using NumPy dot products, ensuring computationally efficient forward passes.
* **Gradient Descent Optimizer**: Engineered the manual calculation of partial derivatives (gradients $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$) to iteratively minimize Mean Squared Error (MSE).
* **Smart Convergence Control**: Programmed a custom dynamic tolerance monitor ($\Delta J < 10^{-4}$). The engine constantly tracks consecutive cost deltas and safely terminates training early once the gradient curve flattens, eliminating thousands of redundant computing loops.

---

## 📈 The Iterative Engineering Process

I documented every stage of the model's growth to capture performance shifts and design updates transparently:

| Iteration Step | Optimization / Fix Applied | Train $R^2$ | Test $R^2$ | Key Engineering Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **1. Baseline** | Raw data split with fixed iteration parameters | 0.5524 | 0.5455 | Discovered data leakage in the validation split loop. |
| **2. Early Stopping**| Added absolute delta evaluation to cost history | 0.5524 | 0.5455 | Automatically converged at step 1064, saving massive CPU cycles. |
| **3. Bug Evacuation**| Rectified inverted ordinal logic maps in `CAEC`/`CALC` | 0.5627 | 0.5627 | Eliminated data noise; achieved exact generalization balance. |
| **4. Feature Expansion**| Introduced interaction variables & squared terms | 0.5627 | 0.5627 | Hit the mathematical "Underfitting Wall" of linear space. |

---

## 🧠 Key Engineering Insights & Discoveries

### 1. The Floating-Point Illusion
During early training runs, my terminal logs showed a flat, unchanging cost value (e.g., `147.55`), yet an exact equality check (`if cost_old == cost_new`) continuously failed to trigger early stopping. I discovered that console formatting strings disguise reality. Deep in system memory, values were fluctuating out past the 10th decimal place. Shifting to an absolute threshold tolerance window (`if delta < tolerance`) successfully resolved the loop freeze.

### 2. Perfect Generalization vs. Structural Boundaries
By Phase 4, the model achieved perfectly synchronized scores on both splits: **Train $R^2$: 0.5627** and **Test $R^2$: 0.5627**. While this proves the architecture has absolutely **zero overfitting**, it highlights a structural plateau. Human physical dimensions and behavioral dependencies are intrinsically non-linear; a pure linear combinations formula mathematically lacks the geometric flexibility to trace complex, multi-dimensional bodily habits perfectly.

---

## 📂 Repository File Structure

* `preprocessing.ipynb`: The complete data preparation pipeline including categorical transformations (Label & One-Hot Encoding), descriptive statistics, and input feature manipulation.
* `model.ipynb`: Contains the manual NumPy linear regression equations, cost tracking arrays, gradient engine, evaluation loops, and performance metric assertions.
* `processed.csv`: Cleaned, scaled, and structured feature space utilized directly by the custom model split.

---

## 🚀 Future Roadmap
* **Algorithmic Transition**: Implement tree-based ensemble frameworks (like Random Forest and Gradient Boosting) to explicitly contrast how non-linear step-boundaries map this health feature space compared to pure linear boundaries.
* **Regularization Matrix**: Integrate custom L1 (Lasso) and L2 (Ridge) penalty variables directly into the scratch gradient cost arrays to observe feature weight suppression behaviors.

```
