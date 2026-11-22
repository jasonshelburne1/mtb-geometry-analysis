"""
Mountain Bike Geometry Analysis
===============================
How do mountain bike geometry measurements relate to one another, and how much
of that variation does wheel size explain?

Examines head tube angle, wheelbase, reach, stack, chainstay length, and rear
suspension travel across 26", 27.5", 29", and Mullet (29" front / 27.5" rear)
non-motorized mountain bikes, using geometry data from MTB-News
(geometrics.mtb-news.de).

Run:    python mtb_geometry_analysis.py
Needs:  pandas, matplotlib, seaborn, statsmodels
Output: summary statistics in the console and charts saved to ./figures/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

DATA_PATH = Path("data/geometrics_mtb-news_de.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# %% 1. Load and inspect --------------------------------------------------------
# The dataset uses semicolons instead of commas to separate columns.
df = pd.read_csv(DATA_PATH, sep=";")
print("Original dataset shape:", df.shape)
print("\nColumns in dataset:")
print(df.columns.tolist())


# %% 2. Filter and clean --------------------------------------------------------
# Keep only non-motorized mountain bikes.
mtb = df[(df["Category"] == "Mountain") & (df["Motorized"] == False)].copy()  # noqa: E712

# Keep the four wheel configurations that are the focus of this project.
wheel_sizes = ["26″", "27,5″ / 650B", "29″", "Mullet 29/27,5"]
mtb = mtb[mtb["Wheel Size"].isin(wheel_sizes)].copy()

# Rename the wheel-size categories so the graphs are easier to read.
mtb["Wheel Size"] = mtb["Wheel Size"].replace({
    "26″": "26",
    "27,5″ / 650B": "27.5",
    "29″": "29",
    "Mullet 29/27,5": "Mullet",
})

# Order in which wheel sizes appear in the graphs.
wheel_order = ["26", "27.5", "29", "Mullet"]

# Convert geometry columns to numeric values (invalid entries become NaN).
geometry_columns = [
    "Head Tube Angle",
    "Chainstay Length",
    "Wheelbase",
    "Reach",
    "Stack",
    "Suspension Travel (rear)",
    "Suspension Travel (front)",
]
for column in geometry_columns:
    mtb[column] = pd.to_numeric(mtb[column], errors="coerce")

print("\nRows after filtering to non-motorized MTBs in the four wheel sizes:", len(mtb))
print("\nMissing values:")
print(mtb[geometry_columns].isnull().sum())

# Remove physically unrealistic geometry values (e.g., a 1 mm wheelbase).
# Rows with missing values in these four columns are also dropped.
mtb = mtb[
    (mtb["Chainstay Length"] >= 350) & (mtb["Chainstay Length"] <= 500)
    & (mtb["Wheelbase"] >= 900) & (mtb["Wheelbase"] <= 1500)
    & (mtb["Reach"] >= 300) & (mtb["Reach"] <= 600)
    & (mtb["Stack"] >= 450) & (mtb["Stack"] <= 750)
].copy()

print("Rows after removing unrealistic or missing geometry:", len(mtb))


# %% 3. Plot 1: Wheel size distribution ----------------------------------------
# Group sizes are very uneven, so comparisons involving 26" bikes are less reliable.
wheel_counts = mtb["Wheel Size"].value_counts().reindex(wheel_order)
print("\nNumber of records by wheel size:")
print(wheel_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=wheel_counts.index, y=wheel_counts.values)
plt.title("Number of Mountain Bike Records by Wheel Size")
plt.xlabel("Wheel Size")
plt.ylabel("Number of Records")
plt.savefig(FIG_DIR / "01_wheel_size_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 4. Summary statistics by wheel size ---------------------------------------
summary = (
    mtb.groupby("Wheel Size", observed=True)[
        ["Head Tube Angle", "Wheelbase", "Reach", "Stack", "Chainstay Length"]
    ]
    .median()
    .reindex(wheel_order)
    .round(1)
)
print("\nMedian geometry by wheel size:")
print(summary)


# %% 5. Plot 2: Head tube angle by wheel size ----------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=mtb, x="Wheel Size", y="Head Tube Angle", order=wheel_order)
plt.title("Head Tube Angle by Wheel Size")
plt.xlabel("Wheel Size")
plt.ylabel("Head Tube Angle (degrees)")
plt.savefig(FIG_DIR / "02_head_tube_angle.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 6. Plot 3: Wheelbase distribution by wheel size ---------------------------
plt.figure(figsize=(8, 5))
sns.violinplot(data=mtb, x="Wheel Size", y="Wheelbase", order=wheel_order)
plt.title("Wheelbase Distribution by Wheel Size")
plt.xlabel("Wheel Size")
plt.ylabel("Wheelbase (mm)")
plt.savefig(FIG_DIR / "03_wheelbase.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 7. Plot 4: Chainstay length vs. wheelbase ---------------------------------
plot_data = mtb.dropna(subset=["Chainstay Length", "Wheelbase"])
corr_chainstay = plot_data["Chainstay Length"].corr(plot_data["Wheelbase"])
print(f"\nCorrelation, chainstay length vs. wheelbase: {corr_chainstay:.2f}")

plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_data, x="Chainstay Length", y="Wheelbase",
                hue="Wheel Size", hue_order=wheel_order, alpha=0.6)
plt.title(f"Chainstay Length vs. Wheelbase (r = {corr_chainstay:.2f})")
plt.xlabel("Chainstay Length (mm)")
plt.ylabel("Wheelbase (mm)")
plt.savefig(FIG_DIR / "04_chainstay_vs_wheelbase.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 8. Plot 5: Reach vs. stack ------------------------------------------------
plot_data = mtb.dropna(subset=["Reach", "Stack"])
corr_reach = plot_data["Reach"].corr(plot_data["Stack"])
print(f"Correlation, reach vs. stack: {corr_reach:.2f}")

plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_data, x="Reach", y="Stack",
                hue="Wheel Size", hue_order=wheel_order, alpha=0.6)
plt.title(f"Reach vs. Stack by Wheel Size (r = {corr_reach:.2f})")
plt.xlabel("Reach (mm)")
plt.ylabel("Stack (mm)")
plt.savefig(FIG_DIR / "05_reach_vs_stack.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 9. Plot 6: Rear suspension travel -----------------------------------------
# Rear travel is missing for many bikes (e.g., hardtails), so only bikes with data are shown.
plot_data = mtb.dropna(subset=["Suspension Travel (rear)"])
print("\nBikes with rear travel data by wheel size:")
print(plot_data["Wheel Size"].value_counts().reindex(wheel_order))

plt.figure(figsize=(8, 5))
sns.boxplot(data=plot_data, x="Wheel Size", y="Suspension Travel (rear)", order=wheel_order)
plt.title("Rear Suspension Travel by Wheel Size")
plt.xlabel("Wheel Size")
plt.ylabel("Rear Suspension Travel (mm)")
plt.savefig(FIG_DIR / "06_rear_travel.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 10. Regression: does wheel size matter once other geometry is controlled? --
# Plots suggest longer wheelbases on larger-wheel bikes, but those bikes also tend to have
# longer reach, longer chainstays, and slacker head angles. A multiple linear regression
# separates these effects:
#
#   Wheelbase = b0 + b1*Reach + b2*Stack + b3*Chainstay + b4*HeadTubeAngle + wheel-size effects
#
# Wheel size is a category, so each coefficient is the difference vs. 29" bikes (the
# largest group) with the other measurements held constant.
#
# Caution: wheelbase is partly determined by these measurements by construction, so a high
# R-squared is expected. The interesting question is whether wheel size adds anything.
reg = mtb.dropna(subset=["Head Tube Angle"]).copy()
reg = reg.rename(columns={
    "Head Tube Angle": "Head_Tube_Angle",
    "Chainstay Length": "Chainstay_Length",
    "Suspension Travel (rear)": "Rear_Travel",
})
reg["Wheel"] = pd.Categorical(reg["Wheel Size"], categories=["29", "26", "27.5", "Mullet"])

# Each bike appears once per frame size, so rows from the same bike are not independent.
# Standard errors are clustered by bike (brand + model + year) to account for this.
reg["bike_id"] = reg["Brand"] + " " + reg["Model"] + " " + reg["Year"].astype(str)

full_model = smf.ols(
    "Wheelbase ~ Reach + Stack + Chainstay_Length + Head_Tube_Angle + C(Wheel)",
    data=reg,
).fit(cov_type="cluster", cov_kwds={"groups": reg["bike_id"]})

# Same model without wheel size, to see how much wheel size adds.
base_model = smf.ols(
    "Wheelbase ~ Reach + Stack + Chainstay_Length + Head_Tube_Angle", data=reg
).fit()

print(f"\nRegression sample: {len(reg):,} records from {reg['bike_id'].nunique():,} bikes")
print(full_model.summary().tables[1])
print(f"R-squared, full model:           {full_model.rsquared:.3f}")
print(f"R-squared, without wheel size:   {base_model.rsquared:.3f}")
print(f"Added by wheel size:             {full_model.rsquared - base_model.rsquared:.4f}")

# Actual vs. predicted wheelbase
plt.figure(figsize=(7, 6))
plt.scatter(full_model.fittedvalues, reg["Wheelbase"], alpha=0.3, s=12)
lims = [reg["Wheelbase"].min(), reg["Wheelbase"].max()]
plt.plot(lims, lims, color="red", linewidth=1)
plt.title(f"Actual vs. Predicted Wheelbase (R² = {full_model.rsquared:.3f})")
plt.xlabel("Predicted wheelbase (mm)")
plt.ylabel("Actual wheelbase (mm)")
plt.savefig(FIG_DIR / "07_regression_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()


# %% 11. Regression: what explains head tube angle? ----------------------------
# Wheelbase was almost fully explained by other measurements. Head tube angle is different:
# it is a design choice (steeper = quicker steering, slacker = more stable at speed).
# Wheelbase is left out of these models because it is itself partly determined by head tube
# angle, which would make the model circular.
#
# Model A: reach, stack, chainstay length, and wheel size (all records)
# Model B: adds rear suspension travel as a rough proxy for the bike's intended use
#          (more travel = more gravity-oriented). Only bikes with rear travel data are
#          included, so hardtails are excluded.
#
# To compare A and B fairly, both are also fit on the same bikes (rows with rear travel).
def fit_clustered(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["bike_id"]}
    )


formula_a = "Head_Tube_Angle ~ Reach + Stack + Chainstay_Length + C(Wheel)"
formula_b = "Head_Tube_Angle ~ Reach + Stack + Chainstay_Length + Rear_Travel + C(Wheel)"

model_a_all = fit_clustered(formula_a, reg)
reg_travel = reg.dropna(subset=["Rear_Travel"])
model_a = fit_clustered(formula_a, reg_travel)
model_b = fit_clustered(formula_b, reg_travel)

print(f"\nModel A, all records ({len(reg):,}): R-squared = {model_a_all.rsquared:.3f}")
print(f"\nSame-sample comparison ({len(reg_travel):,} records with rear travel, "
      f"{reg_travel['bike_id'].nunique():,} bikes)")
print(f"Model A (no travel): R-squared = {model_a.rsquared:.3f}")
print(f"Model B (+ travel):  R-squared = {model_b.rsquared:.3f}")
print("\nModel B coefficients:")
print(model_b.summary().tables[1])

# Mullet vs. 29" head tube angle difference, before and after accounting for travel
mullet = "C(Wheel)[T.Mullet]"
print(f"\nMullet vs. 29\" head angle difference (degrees): "
      f"{model_a.params[mullet]:.2f} without travel, {model_b.params[mullet]:.2f} with travel")


# %% 12. Dashboard: four key plots in one figure -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mountain Bike Geometry by Wheel Size", fontsize=16, fontweight="bold")

sns.boxplot(data=mtb, x="Wheel Size", y="Head Tube Angle", order=wheel_order, ax=axes[0, 0])
axes[0, 0].set(title="Head Tube Angle", xlabel="Wheel Size", ylabel="Degrees")

sns.violinplot(data=mtb, x="Wheel Size", y="Wheelbase", order=wheel_order, ax=axes[0, 1])
axes[0, 1].set(title="Wheelbase", xlabel="Wheel Size", ylabel="Wheelbase (mm)")

data_cw = mtb.dropna(subset=["Chainstay Length", "Wheelbase"])
sns.scatterplot(data=data_cw, x="Chainstay Length", y="Wheelbase", hue="Wheel Size",
                hue_order=wheel_order, alpha=0.5, ax=axes[1, 0])
axes[1, 0].set(title=f"Chainstay Length vs. Wheelbase (r = {corr_chainstay:.2f})",
               xlabel="Chainstay Length (mm)", ylabel="Wheelbase (mm)")

data_rs = mtb.dropna(subset=["Reach", "Stack"])
sns.scatterplot(data=data_rs, x="Reach", y="Stack", hue="Wheel Size",
                hue_order=wheel_order, alpha=0.5, ax=axes[1, 1])
axes[1, 1].set(title=f"Reach vs. Stack (r = {corr_reach:.2f})", xlabel="Reach (mm)", ylabel="Stack (mm)")

plt.tight_layout()
plt.savefig(FIG_DIR / "08_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nCharts saved to ./figures/")
