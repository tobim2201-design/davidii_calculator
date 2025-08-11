import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. CSV laden
train_df = pd.read_csv("bat_measurements_extracted.csv")  # hindfoot, tibia, species
X_train = train_df[['hindfoot_mm', 'tibia_mm']]
y_train = train_df['species']

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# --- App UI ---
st.title("*Myotis davidii* oder *Myotis mystacinus*")
st.write("Artvorhersage durch LDA mittels 2 Parametern.")

# 2. Eingaben
hindfoot_new = st.number_input("Hinterfußlänge (mm)", min_value=0.0, step=0.1)
tibia_new = st.number_input("Tibialänge (mm)", min_value=0.0, step=0.1)

# Fester zweiter Punkt (Dreieck, blau)
ref_point_hindfoot = 35.0
ref_point_tibia = 40.0

if st.button("Vorhersage starten"):
    X_new = [[hindfoot_new, tibia_new]]

    # 3. Vorhersage
    pred_species = lda.predict(X_new)[0]
    pred_prob = lda.predict_proba(X_new)[0]

    st.subheader(f"Vorhergesagte Art: **{pred_species}**")
    for s, p in zip(lda.classes_, pred_prob):
        st.write(f"{s}: {p:.1%}")

    # 4. Plot erstellen
    fig, ax = plt.subplots()

    # Trainingspunkte
    for species_name, group in train_df.groupby('species'):
        ax.scatter(group['hindfoot_mm'], group['tibia_mm'], label=species_name, alpha=0.6)

    # Neuer eingegebener Punkt – roter Kreis
    ax.scatter(hindfoot_new, tibia_new,
               color='red', marker='o', edgecolor='black', s=120,
               label='Neue Messung', zorder=5)

    # Fester Referenzpunkt – blaues Dreieck
    ax.scatter(ref_point_hindfoot, ref_point_tibia,
               color='blue', marker='^', edgecolor='black', s=120,
               label='Referenzpunkt', zorder=5)

    # Textlabel für den neuen Punkt
    ax.text(hindfoot_new + 0.1, tibia_new,
            f"{pred_species}\n{max(pred_prob):.1%}",
            fontsize=10, fontweight='bold', color='red')

    ax.set_xlabel("Hind foot length [mm]")
    ax.set_ylabel("Tibia length [mm]")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, loc='best')

    st.pyplot(fig)

