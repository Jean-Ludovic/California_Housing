import streamlit as st
from src.data_loader import load_data
from src.analysis import (
    price_distribution,
    boxplot_price_by_location,
    correlation_heatmap,
    scatter_plot,
    multi_histograms
)

# Config
st.set_page_config(page_title="Analyse des prix de logements en Californie", layout="wide")

# Introduction
st.title(" Bienvenue dans l'Analyse Exploratoire de Données de California Housing")
st.markdown("""
Bienvenue dans cette application d'exploration de données !  
Nous allons **voyager ensemble** à travers le jeu de données *California Housing Prices*  
pour découvrir ce qui influence réellement le **prix des logements** en Californie .

### Que voulons-nous faire ?
-  Voir un **aperçu du dataset**
-  Explorer des **visualisations statistiques**
-  Identifier les **facteurs clés** qui influencent les prix
""")

# Charger les données
df = load_data()

# Navigation par menu ou boutons
choix = st.radio("Où souhaitez-vous commencer ?", ["Aperçu du Dataset", "Analyse Statistique", "Visualisations" , "Rapport Final"])

# Section 1 — Aperçu
if choix == "Aperçu du Dataset":
    st.subheader(" Aperçu du Dataset")
    st.write(df.head())
    st.write("Dimensions :", df.shape)
    st.write("Colonnes :", list(df.columns))
    

# Section 2 — Statistiques
elif choix == "Analyse Statistique":
    st.subheader(" Statistiques Descriptives")
    st.dataframe(df.describe())

# Section 3 — Visualisations
elif choix == "Visualisations":
    st.subheader(" Visualisations Graphiques")

    # 📌 1. Distribution des prix
    st.markdown("###  Distribution des prix des logements")
    st.markdown("""
    Explorons d'abord comment les prix des maisons sont répartis.  
    Nous allons utiliser un **diagramme à barres** (histogramme) qui montre **combien de maisons** coûtent dans certaines **tranches de prix**.

    Plus une barre est haute, plus il y a de maisons qui coûtent ce prix.  
    Cela nous aide à voir si la majorité des maisons sont chères ou abordables.
    """)
    st.pyplot(price_distribution(df))

    #  2. Boxplot selon la localisation
    st.markdown("###  Prix en fonction de la proximité à l'océan")
    st.markdown("""
    Maintenant, voyons comment les prix changent selon la **localisation géographique**, en particulier si la maison est proche de l'océan .

    Le **boxplot** ci-dessous montre la **médiane** (la ligne au milieu), les **valeurs extrêmes**, et les **variations** de prix pour chaque région.  
    C’est comme si on mettait toutes les maisons d’un groupe dans une boîte pour comparer leur valeur.
    """)
    st.pyplot(boxplot_price_by_location(df))

    #  3. Heatmap des corrélations
    st.markdown("###  Corrélation entre les caractéristiques")
    st.markdown("""
    Observons maintenant quelles caractéristiques influencent **le plus fortement** le prix des maisons.  
    Cette **carte de chaleur** (heatmap) compare chaque chiffre du tableau : quand deux chiffres montent ou descendent ensemble, ils sont corrélés .

    Plus la couleur est **rouge**, plus les deux variables sont liées.  
    Par exemple, si la **surface** augmente, est-ce que le **prix** augmente aussi ?
    """)
    st.pyplot(correlation_heatmap(df))

    #  4. Scatter Plot interactif
    st.markdown("###  Nuage de points interactif")
    st.markdown("""
    Un **nuage de points** (scatter plot) permet de **voir la relation entre deux variables**.  
    Tu choisis une variable à gauche, et on regarde comment le **prix** varie en fonction de cette variable.

    Chaque **point** représente une maison.  
    Si les points montent ensemble, c'est probablement une bonne corrélation !
    """)
    x_var = st.selectbox("Choisissez une variable explicative :", df.select_dtypes(include='number').columns)
    st.pyplot(scatter_plot(df, x=x_var, y="median_house_value"))

    #  5. Histogrammes multiples
    st.markdown("###  Distribution de plusieurs caractéristiques")
    st.markdown("""
    Pour terminer, explorons comment **d'autres caractéristiques** comme l'âge, le nombre de chambres, ou le nombre de familles dans une zone, sont réparties.

    Chaque graphique est comme un **compteur** : plus il est haut, plus cette valeur est fréquente.
    Cela nous permet de mieux connaître les **profils typiques** des logements.
    """)
    st.pyplot(multi_histograms(df))

elif choix == "Rapport Final":
    st.subheader(" Rapport Final : Résumé de l'analyse")

    st.markdown("""
    Ce rapport résume les principales observations tirées de notre analyse exploratoire du dataset *California Housing Prices*.  
    Les visualisations précédentes nous ont permis d'identifier les facteurs ayant le plus d'impact sur les prix des logements.

    ### 1.  Distribution des prix

    La majorité des logements ont un prix médian situé entre **100 000 $ et 250 000 $**,  
    ce qui montre un marché encore **abordable dans de nombreuses zones**.  
    Toutefois, une queue de distribution vers la droite (longue traîne) indique l’existence de **logements très chers** dans certaines régions.

    ### 2.  Influence de la localisation (proximité de l'océan)

    Les maisons proches de la côte ("NEAR OCEAN" ou "ISLAND") présentent des prix **nettement plus élevés**.  
    Cela confirme que la **proximité à l’eau est un facteur premium** dans l’évaluation immobilière.

    ### 3.  Corrélations

    Les variables **positivement corrélées** au prix des logements sont :
    - `median_income` (revenu médian des habitants)
    - `total_rooms` et `housing_median_age` dans une moindre mesure

    En revanche, la **densité de population ou le nombre de ménages** sont peu ou négativement corrélés.

    ### 4.  Relations individuelles (nuages de points)

    Une relation **quasi linéaire** entre le **revenu médian** (`median_income`) et le **prix des logements** a été observée.  
    Plus les habitants gagnent bien leur vie, plus les logements coûtent cher – ce qui est attendu.

    ### 5.  Autres caractéristiques

    - La majorité des maisons ont été construites **entre 20 et 40 ans** en arrière.
    - La plupart des logements contiennent entre **4 et 6 pièces**.
    - Les zones les plus peuplées n’ont pas forcément les logements les plus chers.

    ---

    ###  Conclusion

    Le **revenu médian** et la **localisation géographique** sont les deux **facteurs les plus déterminants** dans le prix d’un logement en Californie.  
    Les agences immobilières devraient **concentrer leurs efforts marketing** dans les zones côtières à revenu élevé, où la valeur immobilière est maximale.
    """)

    report_text = """
             Rapport Final : Résumé de l'analyse

            1. Distribution des prix
            La majorité des logements coûtent entre 100 000 $ et 250 000 $.

            2. Proximité à l'océan
            Les logements près de l’océan sont nettement plus chers.

            3. Corrélations
            Le revenu médian est fortement lié au prix des maisons.

            4. Nuages de points
            On observe une relation linéaire entre revenu médian et prix.

            5. Autres facteurs
            Les maisons ont en moyenne 4 à 6 pièces et ont été construites il y a 20-40 ans.

            Conclusion
            Les revenus et la localisation sont les facteurs les plus influents.
            """ 
    

