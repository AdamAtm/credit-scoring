README (Explication détaillée) :
Étapes pour lancer les scripts
Cloner le projet : La première étape consiste à cloner le projet sur votre machine locale. Cela peut se faire en utilisant la commande git clone. Assurez-vous d’avoir git installé sur votre système, puis exécutez :


git clone <lien_du_dépôt>

Installer les dépendances : Le projet utilise des bibliothèques Python externes telles que pandas, scikit-learn, Flask, et dash. Ces bibliothèques sont répertoriées dans un fichier requirements.txt. Pour les installer toutes à la fois, vous pouvez utiliser la commande suivante :


Lancer l'API Flask : Une fois les dépendances installées, l’API Flask, qui permet de faire des prédictions à partir de l'identifiant du client, peut être démarrée. Utilisez cette commande :


python app.py
Cela démarrera un serveur local accessible à l’adresse http://127.0.0.1:5000/predict. Vous pouvez ensuite envoyer des requêtes POST à cette URL avec les informations client sous forme JSON pour obtenir une prédiction et une probabilité associée.

Tester l'API : Pour tester l’API, vous pouvez utiliser un client comme Postman ou une simple requête Python avec la bibliothèque requests. Voici un exemple de requête POST en Python :


import requests
client_data = {"SK_ID_CURR": 123456}
response = requests.post('http://127.0.0.1:5000/predict', json=client_data)
print(response.json())
Lancer le tableau de bord interactif (Dash) : Le tableau de bord interactif a été construit avec Dash et permet d'explorer les données des clients et de visualiser les prédictions du modèle. Pour démarrer le tableau de bord, exécutez :


python dashboard.py
Le tableau de bord sera accessible via votre navigateur à l'adresse http://127.0.0.1:8050. Il propose différentes visualisations interactives permettant aux conseillers de mieux comprendre les décisions du modèle et les informations associées à chaque client.

Notebook Python (Explication détaillée)
Problèmes rencontrés sur le jeu de données :
Valeurs manquantes :

Description du problème : Le dataset contient de nombreuses valeurs manquantes, surtout dans les colonnes catégoriques et numériques. Cela peut affecter la qualité des prédictions si elles ne sont pas traitées correctement.
Solution : Pour gérer cela, nous avons utilisé des méthodes simples et robustes de remplissage :
Pour les colonnes numériques, nous avons rempli les valeurs manquantes avec la médiane. Cette approche est moins sensible aux valeurs aberrantes que la moyenne.
Pour les colonnes catégoriques, nous avons utilisé le mode, c'est-à-dire la valeur la plus fréquente, afin de maintenir la cohérence des données.
Déséquilibre des classes :

Description du problème : La variable cible (indiquant si un client fait défaut ou non sur son crédit) est très déséquilibrée. En général, les clients non-défaillants sont beaucoup plus nombreux que les défaillants. Cela pourrait entraîner un modèle biaisé en faveur de la classe majoritaire.
Solution envisagée : Pour contrer ce déséquilibre, nous avons prévu d'utiliser SMOTE (Synthetic Minority Over-sampling Technique) qui permet de créer des exemples synthétiques pour la classe minoritaire. Cependant, pour cette première version rapide du modèle, nous avons opté pour un simple train-test split sans équilibrage afin de tester le flux global.
Nettoyage des données :
Encodage des variables catégoriques :

Certaines variables du dataset sont catégoriques, comme CODE_GENDER, NAME_CONTRACT_TYPE, etc. Pour les transformer en données compréhensibles par notre modèle de machine learning, nous avons utilisé plusieurs techniques d'encodage :
Pour les variables binaires (celles avec seulement deux catégories, comme CODE_GENDER), nous avons utilisé LabelEncoder, ce qui transforme les catégories en 0 et 1.
Pour les autres variables avec plus de deux catégories, nous avons appliqué la technique de OneHotEncoding. Cela crée des colonnes indicatrices où chaque catégorie devient une nouvelle colonne avec une valeur binaire (0 ou 1).
Création de nouvelles features :

Nous avons également ajouté de nouvelles colonnes dérivées de certaines informations financières et comportementales :
CREDIT_INCOME_PERCENT : le rapport entre le montant du crédit et le revenu total du client.
ANNUITY_INCOME_PERCENT : le rapport entre l'annuité du crédit et le revenu total.
CREDIT_TERM : la durée du crédit, obtenue en divisant l'annuité par le montant total du crédit.
DAYS_EMPLOYED_PERCENT : le pourcentage de jours travaillés par rapport à l'âge du client, calculé avec DAYS_EMPLOYED et DAYS_BIRTH.
Modélisation (du preprocessing à la prédiction) :
Prétraitement des données :

Après avoir nettoyé et enrichi le dataset, nous avons normalisé les variables numériques à l'aide de StandardScaler. La normalisation est essentielle dans les algorithmes de régression logistique pour s'assurer que toutes les variables sont sur une échelle similaire, ce qui améliore les performances du modèle.
Entraînement du modèle :

Nous avons utilisé un modèle de régression logistique pour notre première version du scoring de crédit. Ce modèle est simple, interprétable et rapide à entraîner.
Pour entraîner le modèle, nous avons divisé les données en un ensemble d'entraînement (80 %) et un ensemble de validation (20 %). La régression logistique a été ajustée à l’aide de la bibliothèque Scikit-learn avec des hyperparamètres standards.
Évaluation du modèle :

Pour évaluer la performance du modèle, nous avons calculé le score ROC-AUC sur les données de validation. Le ROC-AUC est une mesure utile dans les problèmes de classification, car il évalue la capacité du modèle à distinguer entre les classes positives (défaillants) et négatives (non-défaillants).
Résultat initial : Un score ROC-AUC de 0.68 a été obtenu, ce qui est un point de départ raisonnable pour un modèle rapide.
Enregistrement du modèle et du scaler :

Pour rendre le modèle réutilisable dans une API Flask, nous avons enregistré à la fois le modèle et le scaler avec Pickle. Cela permet de charger ces objets plus tard sans avoir à réentraîner le modèle à chaque requête.
Points à améliorer dans la prochaine version :
Utilisation de SMOTE : Pour mieux gérer le déséquilibre des classes et améliorer la prédiction des clients défaillants.
Optimisation des hyperparamètres : Utiliser des techniques d'optimisation comme Hyperopt pour ajuster les hyperparamètres du modèle et améliorer ses performances.
Matrice de coût personnalisée : Incorporer une matrice de coût qui reflète l’impact financier réel des faux positifs et des faux négatifs pour la banque, afin d’adapter l'optimisation du modèle à la réalité métier.