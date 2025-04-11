import shap
import matplotlib.pyplot as plt

def explain(model, X):
    # Convert all columns to float64
    X_numeric = X.astype("float64")

    # Create explainer and generate SHAP values
    explainer = shap.Explainer(model, X_numeric)
    shap_values = explainer(X_numeric, check_additivity=False)

    # Save summary plot
    shap.summary_plot(shap_values, X_numeric, show=False)
    shap_img_path = "shap_summary_plot.png"
    plt.savefig(shap_img_path)
    plt.close()

    return shap_img_path