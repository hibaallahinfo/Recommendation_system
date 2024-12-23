from flask import Flask, session, render_template, request, redirect, url_for, flash, jsonify
import os
from passlib.context import CryptContext
import pandas as pd
import numpy as np
import bcrypt
from recommender import find_top_similar_items, get_embedding, img_path, model  # Import des fonctions
from popularity_recommender import PopularityRecommender
from datetime import datetime

app = Flask(__name__)
app.secret_key = "votre_clé_secrète"

# Chemin vers le fichier CSV pour les utilisateurs
USER_FILE = 'user.csv'
# Charger les données des produits et les embeddings précalculés
PRODUCT_FILE = 'static/data/styles.csv'
EMBEDDINGS_FILE = 'static/data/embeddings5000.npy'
#updated_embeddings = np.load("static/data/updated_embeddings.npy")
df = pd.read_csv(PRODUCT_FILE)
embeddings = np.load(EMBEDDINGS_FILE)
#print("Données chargées avec succès !")

# Créer un contexte pour le hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256", "argon2"], deprecated="auto")

# Fonction pour hacher le mot de passe
def hash_password(password):
    return pwd_context.hash(password)

# Route d'inscription
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Récupérer les données du formulaire
        fullname = request.form['fullname']
        dob = request.form['dob']
        gender = request.form['gender']
        email = request.form['email']
        password = request.form['password']
        favorite_colors = request.form.getlist('favorite_colors[]')  # Liste des couleurs préférées
        favorite_category = request.form['favorite_category']
        
        # Hacher le mot de passe avant de l'enregistrer
        hashed_password = hash_password(password)
        
        # Lire ou créer le fichier CSV
        if os.path.exists(USER_FILE):
            # Lire le fichier existant
            df = pd.read_csv(USER_FILE)
            # Déterminer le prochain ID en fonction du dernier ID existant
            next_id = df['id'].max() + 1
        else:
            # Créer un DataFrame vide avec une colonne pour l'ID
            df = pd.DataFrame(columns=['id', 'fullname', 'dob', 'gender', 'email', 'password', 'favorite_colors', 'favorite_category'])
            next_id = 1  # Premier ID

        # Créer une nouvelle ligne à ajouter au CSV
        new_user = {
            'id': next_id,
            'fullname': fullname,
            'dob': dob,
            'gender': gender,
            'email': email,
            'password': hashed_password,  # Utiliser le mot de passe haché
            'favorite_colors': ', '.join(favorite_colors),  # Convertir la liste de couleurs en chaîne
            'favorite_category': favorite_category
        }
        
        # Ajouter le nouvel utilisateur au DataFrame
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        df.to_csv(USER_FILE, index=False)  # Sauvegarder le DataFrame dans le fichier CSV

        flash("Inscription réussie !", "success")
        return redirect(url_for('login'))  # Rediriger vers la page de connexion après l'inscription

    return render_template("register.html")  # Afficher le formulaire d'inscription


@app.route('/products', methods=['GET'])
def products():
    # Récupérer les paramètres des filtres
    filters = {
        'gender': request.args.get('gender'),
        'masterCategory': request.args.get('masterCategory'),
        'subCategory': request.args.get('subCategory'),
        'articleType': request.args.get('articleType'),
        'baseColour': request.args.get('baseColour'),
        'season': request.args.get('season'),
        'year': request.args.get('year'),
        'usage': request.args.get('usage'),
    }

    # Charger les données
    df = pd.read_csv(PRODUCT_FILE)

    # Appliquer les filtres dynamiquement
    for key, value in filters.items():
        if value:
            if key == 'year':
                df = df[df[key] == int(value)]
            else:
                df = df[df[key] == value]

    # Pagination
    page = int(request.args.get('page', 1))
    per_page = 10
    total_products = len(df)
    total_pages = (total_products // per_page) + (1 if total_products % per_page else 0)

    # Sélectionner les produits pour la page actuelle
    start = (page - 1) * per_page
    end = start + per_page
    products_page = df.iloc[start:end]

    # Calculer l'index global pour chaque produit
    products_page['global_id'] = range(start, end)

    # Convertir les données filtrées et paginées en dictionnaire
    products = products_page.to_dict(orient='records')

    return render_template('products.html', products=products, page=page, total_pages=total_pages, filters=filters)


@app.route('/product_details/<string:product_name>', methods=['GET'])
def product_details(product_name):
    # Convertir les espaces encodés (%20) en espaces, mais garder les tirets
    formatted_product_name = product_name.replace('%20', ' ').title()
    
    # Recherche du produit dans le DataFrame
    product_row = df[df['productDisplayName'].str.lower() == formatted_product_name.lower()]
    
    if product_row.empty:
        return jsonify({"error": "Produit non trouvé"}), 404
    
    product_index = product_row.index[0]
    target_embedding = embeddings[product_index].reshape(1, -1)
    similar_indices, _ = find_top_similar_items(product_index, embeddings, top_n=10)

    product = product_row.iloc[0].to_dict()
    similar_products = df.iloc[similar_indices].to_dict(orient='records')

    return render_template('product_details.html', product=product, similar_products=similar_products)


@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        # Sauvegarder les préférences de l'utilisateur
        user_preferences.update(request.json)
        print(f"Préférences enregistrées : {user_preferences}")
        return jsonify({"status": "success"})
    
    return render_template('preferences.html')

import bcrypt

@app.route("/", methods=["GET", "POST"])
def login():
    try:
        users_df = pd.read_csv(USER_FILE)
        print("Données utilisateurs chargées :", users_df.head())
    except FileNotFoundError:
        print("Fichier utilisateur introuvable.")
        return redirect(url_for('login'))

    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')

        print(f"Email saisi : {email}")
        print(f"Mot de passe saisi : {password}")

        # Rechercher l'utilisateur
        user = users_df[users_df['email'].str.lower() == email.lower()].reset_index()
        print("Utilisateur trouvé :", user)

        if not user.empty:
            hashed_password = user.loc[0, 'password']
            # Vérifier le mot de passe
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                session['user_id'] = int(user.loc[0, 'id'])
                session['user_name'] = user.loc[0, 'fullname']
                print("Connexion réussie.")
                return redirect(url_for('dashboard'))
            else:
                print("Mot de passe incorrect.")
        else:
            print("Utilisateur non trouvé.")

    return render_template("login.html")




# Tableau de bord après connexion
@app.route("/dashboard")
def dashboard():
    # Vérifiez si l'utilisateur est connecté
    if 'user_id' not in session:
        print("Veuillez vous connecter pour accéder au tableau de bord.", "danger")
        return redirect(url_for('login'))
    print("Session actuelle :", session)  # Ajoutez cette ligne pour déboguer
    # Récupérez les informations de l'utilisateur à partir de la session
    user_id = session.get('user_id')
    user_name = session.get('user_name')
    
    return render_template("dashboard.html", user_id=user_id, user_name=user_name)



@app.route('/product_json/<string:product_name>')
def product_json_by_name(product_name):
    """
    Renvoie les détails du produit et ses produits similaires au format JSON 
    en utilisant productDisplayName.
    """
    # Recherche de l'index correspondant au productDisplayName
    product_row = df[df['productDisplayName'] == product_name]
    
    if product_row.empty:
        return jsonify({"error": "Produit non trouvé"}), 404
    
    product_index = product_row.index[0]
    target_embedding = embeddings[product_index].reshape(1, -1)
    similar_indices, _ = find_top_similar_items(product_index, embeddings, top_n=10)
    
    product = product_row.iloc[0].to_dict()
    similar_products = df.iloc[similar_indices].to_dict(orient='records')
    
    return jsonify({
        'product': product,
        'similar_products': similar_products
    })


#print("Taille des embeddings:", embeddings.shape)

# Route pour les produits populaires avec filtres
@app.route('/popular', methods=['GET'])
def popular_products():
    """
    Route pour afficher les produits populaires avec filtres optionnels.
    """
    # Récupérer les paramètres de filtres via la requête GET
    gender = request.args.get('gender')  # Filtre par genre
    category = request.args.get('masterCategory')  # Filtre par catégorie principale
    sub_category = request.args.get('subCategory')  # Filtre par sous-catégorie
    color = request.args.get('baseColour')  # Filtre par couleur
    n = request.args.get('n', default=30, type=int)  # Nombre de produits à afficher

    # Initialiser le système de recommandation basé sur la popularité
    recommender = PopularityRecommender(df)

    # Obtenir les produits populaires en appliquant les filtres
    top_products = recommender.get_top_products(
        gender=gender, 
        category=category, 
        sub_category=sub_category, 
        color=color, 
        n=n
    )

    # Convertir en dictionnaire pour l'affichage dans le template
    products = top_products.to_dict(orient='records')

    # Retourner les résultats à la page HTML
    return render_template(
        'popular_products.html', 
        products=products, 
        gender=gender, 
        category=category, 
        sub_category=sub_category, 
        color=color
    )

# Correspondance des mois avec les saisons
SEASON_MAPPING = {
    1: "Winter", 2: "Winter", 12: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
}

@app.route("/personalisation/<int:user_id>")
def personalisation(user_id):
    try:
        # Charger les données utilisateur
        users_df = pd.read_csv(USER_FILE)
        if user_id >= len(users_df):
            flash("Utilisateur introuvable.", "danger")
            return redirect(url_for('login'))
        user = users_df.iloc[user_id]
    except FileNotFoundError:
        flash("Fichier utilisateur introuvable.", "danger")
        return redirect(url_for('login'))

    try:
        # Charger les produits
        products_df = pd.read_csv(PRODUCT_FILE)
    except FileNotFoundError:
        flash("Fichier de produits introuvable.", "danger")
        return redirect(url_for('login'))

    # Préférences utilisateur
    favorite_colors = user['favorite_colors'].split(', ')  # Liste des couleurs
    favorite_category = user['favorite_category']
    gender = user['gender']

    # Déterminer la saison actuelle
    current_month = datetime.now().month
    current_season = SEASON_MAPPING[current_month]

    # Filtrer les produits
    def matches_preferences(row):
        return (
            row['masterCategory'] == favorite_category and
            row['season'] == current_season and
            row['gender'] == gender and
            any(color.strip().lower() in row['baseColour'].lower() for color in favorite_colors)
        )

    # Appliquer le filtre
    recommended_products = products_df[products_df.apply(matches_preferences, axis=1)]

    # Vérification des IDs
    recommended_products['id'] = recommended_products['id'].astype(str)

    return render_template(
        "personalisation.html",
        user=user,
        recommended_products=recommended_products,
        season=current_season
    )


@app.route('/logout')
def logout():
    session.clear()  # Supprime toutes les données de session
    flash("Déconnexion réussie.", "success")
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)