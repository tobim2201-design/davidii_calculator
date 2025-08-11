import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. CSV laden
train_df = pd.read_csv("bat_measurements_extracted.csv")  # hindfoot_mm, tibia_mm, species
X_train = train_df[['hindfoot_mm', 'tibia_mm']]
y_train = train_df['species']

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# --- App UI ---
st.title("*Myotis davidii* oder *Myotis mystacinus*")
st.write("Artvorhersage durch LDA mit zwei Variablen.")

# 2. Eingaben
hindfoot_new = st.number_input("Hinterfußlänge (mm)", min_value=0.0, step=0.1)
tibia_new = st.number_input("Tibialänge (mm)", min_value=0.0, step=0.1)

if st.button("Vorhersage starten"):
    X_new = [[hindfoot_new, tibia_new]]

    # 3. Vorhersage
    pred_species = lda.predict(X_new)[0]
    pred_prob = lda.predict_proba(X_new)[0]
    
    st.subheader(f"Vorhergesagte Art: **{pred_species}**")
    # 4. Plot erstellen
    fig, ax = plt.subplots(figsize=(9,6))

    # Stilzuordnung: passe die Keys an deine exakten species-Strings im CSV an
    style_map = {
        "Myotis_davidii": {"color": "blue", "marker": "^"},   # blaues Dreieck
        "Myotis_mystacinus": {"color": "red", "marker": "o"}  # roter Kreis
    }

    # Trainingspunkte mit individuellem Stil
    for species_name, group in train_df.groupby('species'):
        style = style_map.get(species_name, {"color": "grey", "marker": "o"})
        ax.scatter(
            group['hindfoot_mm'],
            group['tibia_mm'],
            c=style["color"],
            marker=style["marker"],
            edgecolors='black',
            linewidths=0.5,
            s=80,
            alpha=0.8,
            label=species_name
        )

    # Neuer eingegebener Punkt – großer roter Kreis mit schwarzem Rand (sichtbar)
    ax.scatter(
        hindfoot_new,
        tibia_new,
        c='green',
        marker='D',
        edgecolors='black',
        s=180,
        label='Neue Messung',
        zorder=5
    )

    # Beschriftung neben dem neuen Punkt
    ax.text(hindfoot_new + 0.08, tibia_new,
            f"{pred_species}\n{max(pred_prob):.1%}",
            fontsize=12, fontweight='bold', color='White',
            bbox ={'facecolor':'darkgreen','alpha':0.8, 'pad':10})
    ax.set_facecolor("grey")
    ax.set_xlabel("Hind foot length [mm]")
    ax.set_ylabel("Tibia length [mm]")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='best')

    plt.tight_layout()
    st.pyplot(fig)














