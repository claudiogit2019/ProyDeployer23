import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# 1.Función para contar la cantidad de películas producidas en un idioma específico

# Cargar el dataset 
dataset_path = "movies_dataset.csv"
df = pd.read_csv(dataset_path)

# Crear una instancia de la aplicación FastAPI
app = FastAPI()


def contar_peliculas_por_idioma(idioma: str):
    cantidad = df[df["original_language"] == idioma].shape[0]
    return cantidad

# Endpoint para obtener la cantidad de películas producidas en un idioma específico
@app.get('/peliculas_idioma/{idioma}')
async def peliculas_idioma(idioma: str):
    cantidad = contar_peliculas_por_idioma(idioma)
    return {'idioma': idioma, 'cantidad': cantidad}  


#2.Función para obtener la duración y el año de una película específica
# Cargar el dataset 
dataset_path = "movies_dataset.csv"
df = pd.read_csv(dataset_path)

# Crear una instancia de la aplicación FastAPI
#app = FastAPI()

# Función para obtener la duración y el año de una película específica
def obtener_duracion_y_anio(pelicula: str):
    pelicula_data = df[df["title"] == pelicula]
    
    if pelicula_data.empty:
        return None, None
    
    duracion = pelicula_data["runtime"].values[0]
    
    try:
        anio = pelicula_data["release_year"].values[0]

    except KeyError:
        anio = None
    
    return duracion, anio

# Endpoint para obtener la duración y el año de una película específica
@app.get('/peliculas_duracion/{pelicula}')
async def peliculas_duracion(pelicula: str):
    duracion, anio = obtener_duracion_y_anio(pelicula)
    
    if duracion is None:
        raise HTTPException(status_code=404, detail="Pelicula no encontrada")
    
    return {'pelicula': pelicula, 'duracion': duracion, 'anio': anio}



#3.Funcion Franquicia
# Cargar el conjunto de datos preprocesado
dataset_path = "movies_dataset.csv"
df = pd.read_csv(dataset_path)

@app.get('/franquicia/')
async def get_franquicia(Franquicia: str):
    franquicia_data = df[df['belongs_to_collection'] == Franquicia]
    
    if franquicia_data.empty:
        return {"error": f"La franquicia {Franquicia} no existe en el dataset."}
    
    num_movies = len(franquicia_data)
    total_revenue = franquicia_data['revenue'].sum()
    average_revenue = franquicia_data['revenue'].mean()
    
    return {
        "message": f"La franquicia {Franquicia} posee {num_movies} películas, una ganancia total de {total_revenue} y una ganancia promedio de {average_revenue}"
    }

#4.películas producidas en el país
# Cargar el conjunto de datos preprocesado
dataset_path = "movies_dataset.csv"
df = pd.read_csv(dataset_path)


@app.get('/peliculas_pais/')
async def get_peliculas_pais(Pais: str):
    pais_data = df[df['original_language'] == Pais]
    
    if pais_data.empty:
        return {"error": f"No se encontraron películas producidas en el país {Pais}."}
    
    num_movies = len(pais_data)
    
    return {
        "message": f"Se produjeron {num_movies} películas en el país {Pais}"
    }


# 5. Productoras
# Cargar el conjunto de datos preprocesado

dataset_path = "movies_dataset11.xlsx"
df = pd.read_excel(dataset_path)



@app.get('/productoras_exitosas/')
async def get_productora_exitosa(Productora: str):
    productora_data = df[df['production_companies'] == Productora]
    
    if productora_data.empty:
        return {"error": f"No se encontraron datos para la productora {Productora}."}
    
    total_revenue = productora_data['revenue'].sum()
    num_movies = len(productora_data)
    
    return {
        "message": f"La productora {Productora} ha tenido un Ganancia de {total_revenue} y ha realizado {num_movies} películas."
    }


#6.Directorres y sus Peliculas
# Cargar los datasets
df_directores = pd.read_csv("credits_dataset2.csv")
df_peliculas = pd.read_csv("movies_dataset.csv")

# Función para obtener la información de un director
def obtener_info_director(nombre_director):
    director_info = df_directores[df_directores['crew_job'] == 'Director']
    director_info = director_info[director_info['crew_name'] == nombre_director]
    return director_info

# Función para obtener la información de las películas de un director
def obtener_peliculas_director(nombre_director):
    director_info = obtener_info_director(nombre_director)
    if director_info.empty:
        return []

    director_id = director_info.iloc[0]['id']
    peliculas = df_peliculas[df_peliculas['crew_ids'].str.contains(str(director_id))]

    peliculas_info = []
    for index, pelicula in peliculas.iterrows():
        pelicula_info = {
            'title': pelicula['title'],
            'release_date': pelicula['release_date'],
            'revenue': pelicula['revenue'],
            'budget': pelicula['budget']
        }
        peliculas_info.append(pelicula_info)
    
    return peliculas_info

# Endpoint para obtener la información del director y sus películas
@app.get('/get_director/{nombre_director}')
async def get_director(nombre_director: str):
    director_info = obtener_info_director(nombre_director)
    if director_info.empty:
        return {'success': False, 'message': 'Director no encontrado'}

    peliculas_info = obtener_peliculas_director(nombre_director)
    
    return {
        'success': True,
        'director_name': nombre_director,
        'peliculas': peliculas_info
    }



# Sistema de Recomendacion 
# Cargar el dataset desde el archivo 
df = pd.read_csv("movies_dataset.csv", nrows=5000)

# Crear la matriz de características TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'].fillna(''))

# Cálculo de la similitud coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recomendacion(titulo):
    idx = df.index[df['title'] == titulo].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movies = df['title'].iloc[movie_indices].tolist()
    
    return recommended_movies

@app.get('/recomendacion/')
async def get_recomendacion(titulo: str):
    recommended_movies = recomendacion(titulo)
    return {"recommended_movies": recommended_movies}
