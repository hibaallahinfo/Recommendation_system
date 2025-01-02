# Fashion Recommendation System with Enhanced Graph-Based Techniques

### Dataset Used

The dataset included in this repository is a compact version (`small`). It contains blurred images but requires less storage space.  

If you prefer a dataset with clearer images, you can use an alternative dataset. This dataset combines the images from the complete version (`big dataset`) with enhanced styles, additional attributes, and pre-extracted embeddings, saving you the effort of recalculating them.  

**Link to the dataset:** [Fashion Emb Final Dataset](https://www.kaggle.com/datasets/salmalaamari/fashion-emb-final-dataset)


This project implements a fashion recommendation system using Flask as framework. The system incorporates the following features and techniques:

- **User Authentication and Registration**: Secure registration and login functionalities with hashed passwords using `bcrypt` and `CryptContext`. User data is stored in a CSV file.
- **Product Filtering and Search**: Users can filter products based on gender, category, color, price, rating, and other attributes. Pagination is implemented for a seamless browsing experience.
- **Enhanced Recommendation Engine**: 
  - Utilizes precomputed embeddings and a graph-based approach to generate product recommendations.
  - Builds an enhanced graph with similarity thresholds, attention weights, and information propagation to refine embeddings.
  - Provides similar items for a selected product based on updated embeddings and similarity metrics.
- **Preference Management**: Supports saving user preferences for personalized experiences.
- **Cart Functionality**: Allows users to add items to their cart for a potential purchase.

Data processing involves `pandas` and `numpy`, with recommendations powered by advanced graph algorithms. The system is designed to provide users with personalized and relevant fashion suggestions while maintaining performance and scalability.



