from flask import Flask, session, render_template, request, redirect, url_for, flash, jsonify
import os
from passlib.context import CryptContext
import pandas as pd
import numpy as np
import bcrypt
from recommender import find_enhanced_similar_items, build_enhanced_graph ,compute_enhanced_attention ,propagate_enhanced_information , img_path, model  # Import des fonctions
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

    # Récupérer les couleurs sélectionnées
    selected_colors = request.args.get('selectedColors', '')  # Une chaîne séparée par des virgules
    selected_colors = selected_colors.split(',') if selected_colors else []

    # Récupérer les plages de prix
    min_price = request.args.get('minPrice', None)
    max_price = request.args.get('maxPrice', None)

    # Charger les données
    df = pd.read_csv(PRODUCT_FILE)

    # Appliquer les filtres dynamiquement
    for key, value in filters.items():
        if value:
            if key == 'year':
                df = df[df[key] == int(value)]
            else:
                df = df[df[key] == value]

    # Filtrer par couleurs
    if selected_colors:
        df = df[df['baseColour'].isin(selected_colors)]

    # Filtrer par plage de prix
    if min_price:
        df = df[df['price'] >= float(min_price)]
    if max_price:
        df = df[df['price'] <= float(max_price)]

    # Pagination
    page = int(request.args.get('page', 1))
    per_page = 12
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




from flask import session

@app.route('/product_details/<string:product_name>', methods=['GET', 'POST'])
def product_details(product_name):
    formatted_product_name = product_name.replace('%20', ' ').title()
    product_row = df[df['productDisplayName'].str.lower() == formatted_product_name.lower()]
    
    if product_row.empty:
        return jsonify({"error": "Produit non trouvé"}), 404
    
    product_index = product_row.index[0]
    embeddings = np.load(EMBEDDINGS_FILE)
    
    # Construire le graphe avec les embeddings
    graph, similarities = build_enhanced_graph(embeddings, min_threshold=0.90, max_neighbors=10)
    
    # Calculer les poids d'attention
    attention_weights = compute_enhanced_attention(graph, embeddings, similarities, temperature=0.5)
    
    # Propagation des informations dans le graphe
    updated_embeddings = propagate_enhanced_information(graph, embeddings, attention_weights, num_iterations=3, decay_factor=0.9)
    
    # Recherche des indices similaires avec le graphe amélioré
    similar_indices, weights = find_enhanced_similar_items(
        product_index, updated_embeddings, similarities, 
        top_n=10, 
        similarity_threshold=0.4
    )
    
    # Récupération des produits similaires
    similar_products = df.iloc[similar_indices].to_dict(orient='records')
    
    # Récupérer le produit actuel
    product = product_row.iloc[0].to_dict()

    # Traitement de l'ajout au panier (méthode POST)
    if request.method == 'POST':
        # Ajouter le produit au panier dans la session
        if 'cart' not in session:
            session['cart'] = []  # Créer un panier s'il n'existe pas
        product_id = product['id']
        
        # Ajouter le produit au panier
        session['cart'].append(product_id)
        
        # Optionnel : Afficher un message de succès
        print('Produit ajouté au panier !', 'success')
    
    # Retourner la page avec les produits similaires et l'état du panier
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

@app.route('/popular', methods=['GET'])
def popular_products():
    """
    Route pour afficher les produits populaires avec filtres optionnels.
    """
    try:
        # Récupérer les paramètres de filtres via la requête GET
        gender = request.args.get('gender')  # Filtre par genre
        category = request.args.get('masterCategory')  # Filtre par catégorie principale
        sub_category = request.args.get('subCategory')  # Filtre par sous-catégorie
        color = request.args.get('baseColour')  # Filtre par couleur
        n = request.args.get('n', default=30, type=int)  # Nombre de produits à afficher
        page = request.args.get('page', default=1, type=int)  # Page actuelle

        # Vérifier que le DataFrame est valide
        if df.empty:
            flash("Les données des produits sont indisponibles.", "danger")
            return render_template('popular_products.html', products=[], gender=gender, category=category, sub_category=sub_category, color=color)

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

        # Calculer le nombre total de produits et le nombre total de pages
        total_products = len(top_products)
        total_pages = (total_products // n) + (1 if total_products % n != 0 else 0)

        # Récupérer les produits pour la page actuelle
        start_idx = (page - 1) * n
        end_idx = start_idx + n
        products_on_page = top_products.iloc[start_idx:end_idx]

        # Convertir les résultats en dictionnaire pour l'affichage dans le template
        products = products_on_page.to_dict(orient='records')

        # Retourner les résultats à la page HTML
        return render_template(
            'popular_products.html',
            products=products,
            gender=gender,
            category=category,
            sub_category=sub_category,
            color=color,
            total_products=total_products,
            total_pages=total_pages,
            current_page=page
        )

    except Exception as e:
        # Gérer les erreurs inattendues
        flash(f"Une erreur est survenue : {str(e)}", "danger")
        return render_template(
            'popular_products.html', 
            products=[], 
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
            flash("User not found.", "danger")
            return redirect(url_for('login'))
        user = users_df.iloc[user_id]
    except FileNotFoundError:
        flash("User file not found.", "danger")
        return redirect(url_for('login'))

    try:
        # Charger les produits
        products_df = pd.read_csv(PRODUCT_FILE)
    except FileNotFoundError:
        flash("Product file not found.", "danger")
        return redirect(url_for('login'))

    # Préférences utilisateur
    favorite_colors = user['favorite_colors'].split(', ')
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

    recommended_products = products_df[products_df.apply(matches_preferences, axis=1)]
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
# Ajout au panier
@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
def add_to_cart(product_id):
    if 'cart' not in session:
        session['cart'] = []

    # Ajouter le produit au panier
    session['cart'].append(product_id)
    flash("Produit ajouté au panier !", "success")
    return redirect(url_for('cart'))



# Supprimer un produit du panier
@app.route('/remove_from_cart/<int:product_id>', methods=['POST'])
def remove_from_cart(product_id):
    if 'cart' in session:
        try:
            session['cart'].remove(product_id)
            flash("Produit supprimé du panier.", "success")
        except ValueError:
            flash("Le produit n'est pas dans votre panier.", "warning")
    return redirect(url_for('cart'))


@app.route('/cart', methods=['GET'])
def cart():
    # Récupérer les IDs des produits dans le panier (stockés dans la session)
    cart_product_ids = session.get('cart', [])
    
    # Si le panier est vide
    if not cart_product_ids:
        return render_template('cart.html', cart_products=[], message="Your cart is empty.", total_price=0)
    
    # Récupérer les informations sur les produits à partir des IDs
    cart_products = df[df['id'].isin(cart_product_ids)].to_dict(orient='records')
    
    # Calculer le prix total du panier
    total_price = sum(product["price"] for product in cart_products)
    
    # Retourner la page du panier avec les produits et le prix total
    return render_template('cart.html', cart_products=cart_products, total_price=total_price)


if __name__ == "__main__":
    app.run(debug=True)