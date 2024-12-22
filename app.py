from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import pandas as pd
import numpy as np
from recommender import find_top_similar_items, get_embedding, img_path, model  # Import des fonctions
from popularity_recommender import PopularityRecommender

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Clé pour les messages flash

# Chemin vers le fichier CSV pour les utilisateurs
USER_FILE = 'user.csv'
# Charger les données des produits et les embeddings précalculés
PRODUCT_FILE = 'static/data/styles.csv'
EMBEDDINGS_FILE = 'static/data/embeddings5000.npy'
updated_embeddings = np.load("static/data/updated_embeddings.npy")
df = pd.read_csv(PRODUCT_FILE)
embeddings = np.load(EMBEDDINGS_FILE)
#print("Données chargées avec succès !")

# Créez le fichier CSV s'il n'existe pas
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["fullname", "dob", "gender", "nationality", "password"]).to_csv(USER_FILE, index=False)


from flask import request

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
    target_embedding = updated_embeddings[product_index].reshape(1, -1)
    similar_indices, _ = find_top_similar_items(product_index, updated_embeddings, top_n=10)

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

# Route d'inscription
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Récupérer les données du formulaire
        fullname = request.form['fullname']
        dob = request.form['dob']
        gender = request.form['gender']
        nationality = request.form['nationality']
        password = request.form['password']

        # Ajouter l'utilisateur dans le fichier CSV
        df = pd.read_csv(USER_FILE)
        if fullname in df['fullname'].values:
            flash("Cet utilisateur existe déjà. Veuillez vous connecter.", "warning")
            return redirect(url_for('login'))

        new_user = pd.DataFrame([[fullname, dob, gender, nationality, password]],
                                columns=["fullname", "dob", "gender", "nationality", "password"])
        new_user.to_csv(USER_FILE, mode='a', header=False, index=False)

        flash("Inscription réussie ! Connectez-vous maintenant.", "success")
        return redirect(url_for('login'))

    return render_template("register.html")


# Route de connexion
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        fullname = request.form['fullname']
        password = request.form['password']

        # Charger les utilisateurs existants depuis le fichier CSV
        df = pd.read_csv(USER_FILE)

        # Vérifier si l'utilisateur existe avec le bon mot de passe
        user = df[(df['fullname'] == fullname) & (df['password'] == password)]

        if not user.empty:  # Si l'utilisateur existe
            flash(f"Bienvenue, {fullname} !", "success")
            return redirect(url_for('dashboard'))  # Redirection vers le tableau de bord
        else:
            flash("Identifiants incorrects. Veuillez réessayer.", "danger")
            return redirect(url_for('login'))  # Retourner à la page de login

    return render_template("login.html")



# Tableau de bord après connexion
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

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
    # Récupérer les paramètres de filtres via la requête GET
    category = request.args.get('masterCategory')  # Filtre par catégorie
    n = request.args.get('n', default=30, type=int)  # Nombre de produits à afficher

    # Initialiser le système de recommandation basé sur la popularité
    recommender = PopularityRecommender(df)

    # Obtenir les produits populaires
    top_products = recommender.get_top_products(category=category, n=n)

    # Convertir en dictionnaire pour l'affichage dans le template
    products = top_products.to_dict(orient='records')

    # Retourner les résultats à la page HTML
    return render_template('popular_products.html', products=products, category=category)

if __name__ == "__main__":
    app.run(debug=True)