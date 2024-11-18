import pandas as pd
import numpy as np
from scipy import stats
import warnings
from scipy.stats import ConstantInputWarning
import seaborn as sns
import matplotlib.pyplot as plt

# Calculo de cardinalidad
def cardinalidad(target_col):
    calculo_cardi = target_col.nunique()
    porcentaje_cardi = target_col.nunique()/len(target_col) * 100
    # Comporbación de cardinalidad
    if (calculo_cardi >= 10) | (porcentaje_cardi >= 30):
        return calculo_cardi
    else:
        raise ValueError("La variable no es numerica")

# Función auxiliar para la comprobación de parámetros
def check_args(df, target_col, umbral_corr, pvalue):
    if not (target_col in df.columns): # compruebo que está en las columnas de DataFrame
        raise ValueError("El parámetro target_col no está en las columnas del dataframe")
    elif not (pd.api.types.is_numeric_dtype(df[target_col])): # compruebo que tiene un valor numérico
        raise ValueError("El parámetro target_col no tiene un valor numérico")
    elif not (cardinalidad(df[target_col])): # compruebo que tiene una cardinalidad < 30
        raise ValueError("El parámetro target_col no tiene suficiente cardinalidad")
    elif not (isinstance(umbral_corr, float) or isinstance(umbral_corr, int)):
        raise ValueError("El parámetro umbral_corr no es de tipo float")
    elif not (0 <= umbral_corr <= 1):
        raise ValueError("El parámetro umbral_corr no está entre el 0 y el 1")
    elif not (isinstance(pvalue, float) or isinstance(pvalue, int)) and not (pvalue == None):
        raise ValueError("El parámetro pvalue no es de tipo float")
    else:
        return True

def describe_df(df):
     """
          Esta función devuelve un dataframe que tiene una columna por cada columna del dataframe original 
          y como filas, los tipos de las columnas, el tanto por ciento de valores nulos o missings, 
          los valores únicos y el porcentaje de cardinalidad.

          Argumentos:
          df (DataFrame): DataFrame a analizar.

          Retorna:
          tipo (DataFrame): una dataframe con las siguiente calculos: tipo dato columa, nulos (%), valores únicos y cardinalildad (%).
     """
  
     for col in df[df.columns]:
     # Almacenamos la información necesaria
          describe = {
               'type': df.dtypes,  # 1.Tipo de dato por columna
               'null (%)': df[col].isnull().mean() * 100,  # 2.valores nulos (%)
               'unique': df.nunique(),  # 3.valores únicos
               'cardinalidad (%)': (df.nunique() / len(df)) * 100  # 4.Cardinalidad (%)
               }

     # Creación un DF
     describe_df = pd.DataFrame(describe)

     # Trabajamos con los porcentajes
     describe_df['null (%)'] = describe_df['null (%)'].round(2)
     describe_df['cardinalidad (%)'] = describe_df['cardinalidad (%)'].round(2)

     # Devolver de manera transpuesta (columnas sean filas)
     return describe_df.T

def tipifica_variables(df,umbral_categoria,umbral_continua):
   """
      Función para clasificar el tipo de variable que hay dentro de un DataFrame.

      Argumentos:
      df (DataFrame): Dataframe a analizar.
      umbral_categoria (int): umbral para definir una variable como categórica.
      umbral_continua (float): umbral para definir una variable como numérica continua, basado en el porcentaje de cardinalidad.
      
      Retorna:
      tipo (DataFrame): un dataframe con dos columnas "nombre_variable", "tipo_sugerido" que tendrá tantas filas como columnas el dataframe.
   """
   # Creamos lista para almacenar el resultado
   sugerencias = []
   for col in df.columns:
      # Calculamos la cardinalidad de valores únicos y porcentaje
      cardinalidad = df[col].nunique()
      porcentaje_cardinalidad = (cardinalidad / len(df)) * 100
      # Determinar el tipo sugerido
      if cardinalidad == 2:
         tipo_sugerido = "binaria"
      elif cardinalidad < umbral_categoria:
         tipo_sugerido = "categórica"
      elif cardinalidad >= umbral_categoria:
         if porcentaje_cardinalidad >= umbral_continua:
            tipo_sugerido = "numerica continua"
         else:
            tipo_sugerido = "numerica discreta"
      # Añadimos la sugerencia a la lista
      sugerencias.append({
         'nombre_variable': col,
         'tipo_sugerido': tipo_sugerido
      })
   df_retorno = pd.DataFrame(sugerencias)
   return df_retorno

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
   """
      Filtra las columnas del Dataframe, para obtener las clasificadas como numericas y que tengan una correlación con "target_col" superior a la pedida en "umbral_corr".
    
      Argumentos:
      df (DataFrame): Dataframe a analizar.
      target_col (DataFrame): Columna objetivo para calcular las correlaciones.
      umbral_corr (float): El umbral de correlación absoluta para seleccionar columnas. Número decimal entre 0 y 1.
      pvalue (float): Opcional, el valor límite del p-valor para la prueba t de una muestra. Si es None, solo se considera la correlación. Número decimal entre 0 y 1.
      
      Retorna:
      tipo (list): Lista con las columnas numéricas del dataframe cuya correlación con la columna "target_col" sea superior en valor absoluto al valor dado por "umbral_corr" y opcionalmente, con la prueba T-test.
   """
   list_res = []
   if check_args(df, target_col, umbral_corr, pvalue):
      df_numerico = df.select_dtypes(include=['number'])
      correlaciones = df_numerico.corr()[target_col]
      col_fil = correlaciones[correlaciones.abs() > umbral_corr].index.tolist()
      col_fil.remove(target_col) # todas las columnas numericas sin la columna target

      if pvalue == None:
         list_res = col_fil
      else:
         for col in col_fil:
            sin_nulos = df[col].dropna()
            t_stat, p_value = stats.ttest_1samp(sin_nulos, 30) # Test t de una muestra del dataframe limpia
            print("p_value->",p_value)
            if 0.05 >= (1-p_value):
               list_res.append(col)
   else:
      print("Los valores introducidos no son correctos")

   return list_res

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
   """
      La función pintará una pairplot del dataframe considerando la columna designada por "target_col" y 
      aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto 
      a "umbral_corr", y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para 
      el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

      Argumentos:
      df (DataFrame): Dataframe a analizar.
      target_col (DataFrame): Columna objetivo para calcular las correlaciones. Valor por defecto = "".
      columns (list): features a relacionar con la columna "target_col". Valor por defecto = [].
      umbral_corr (float): El umbral de correlación absoluta para seleccionar columnas. Número decimal entre 0 y 1. Valor por defecto = 0.
      pvalue (float): Opcional, el valor límite del p-valor para la prueba t de una muestra. Si es None, solo se considera la correlación. Número decimal entre 0 y 1. Valor por defecto = None.

      Retorna:
      tipo (list): los valores de "columns" que cumplan con las condiciones anteriores. 
   """
   if columns:
      list_col = get_features_num_regression(df, target_col, umbral_corr, pvalue) # devuelve una lista con las columnas que buscamos
      plot_cols = list_col + [target_col]
      df_num_filt = df[plot_cols]    
      sns.pairplot(df)
      sns.pairplot(df_num_filt, hue=target_col)
      plt.show()
      
   else:
      columns = df.select_dtypes(include=[np.number]).columns.tolist()
      list_col = get_features_num_regression(df, target_col, umbral_corr, pvalue) # devuelve una lista con las columnas que buscamos
      plot_cols = list_col + [target_col]
      df_num_filt = df[plot_cols]
      sns.pairplot(df)
      sns.pairplot(df_num_filt, hue=target_col)
      plt.show()

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Filtra las columnas categoricas del Dataframe que pueden asociarse a "targe_col".
    Realiza pruebas T-tets o ANOVA para comprobar que tienen relación.

    Argumentos:
    df (DataFrame): Dataframe a analizar.
    target_col (DataFrame): Nombre de la columna numérica objetivo.
    pvalue (float): Valor límite para el p-valor de las pruebas estadísticas. Debe ser un valor entre 0 y 1. Valor por defecto 0.05.

    Retorna:
    features (list): Lista de columnas categóricas del dataframe cuyo test de relación con 'target_col' supere en confianza estadística del test.
    """
    
    # 1. Validación de entrada
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es numérica.")
        return None
    if not (isinstance(pvalue, float) or isinstance(pvalue, int)):
        print("Error: El valor de 'pvalue' no es adecuado. Debe ser un float o entero.")
        return None
    if not (0 <= pvalue <= 1):
        print("Error: El valor de 'pvalue' no es adecuado. Debe ser estar entre 0 y 1.")
        return None
    
    # 2. Validación de cardinalidad 
    calculo_cardi_target = df[target_col].nunique()
    porcentaje_cardi_target = df[target_col].nunique()/len(target_col) * 100
    #Comporbación de cardinalidad
    if not (calculo_cardi_target >= 10) | (porcentaje_cardi_target >= 30):
        print(f"La columna {target_col} no es una variable numerica continua o discreta ")
        return None

    # 3. Comprobación columnas DataFrame son categoricas
    columnas = df.columns
    lista_categoricas = []

    for col in columnas:
        calculo_cardi = df[col].nunique()
        if calculo_cardi < 10:
            lista_categoricas.append(col)
            continue
        else:
            continue
    
    # 3.1 Si no hay columnas categoricas
    if len(lista_categoricas) == 0:
        print("Error: No hay columnas categóricas en el DataFrame.")
        return None
    
    #4. Tratamos Nan en el DataFrame y añadimos las features
    features = []
    for col in lista_categoricas:
        # Eliminamos filas con valores NaN en el DataFrame
        df_sin_nan = df.dropna()
        parametro = len(df_sin_nan[col].unique())
        if parametro <= 1:
            # Columna categórica sin suficiente variación
            continue
        if parametro == 2:
            # Test t de Student si es Binaria
            grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
            stat, p = stats.ttest_ind(*grupo_cat, equal_var=False)
        else:
            # ANOVA si hay más de dos categorías
            grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
            stat, p = stats.f_oneway(*grupo_cat)
            warnings.filterwarnings("ignore", category= ConstantInputWarning) #Quitar aviso de baja variacion, ya lo revisamos en parametro ==1 y == 2

        if p < pvalue:
            features.append(col)
    return features if features else None

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
        La función pintará los histogramas agrupados de la variable "target_col" para cada uno de los valores de las variables categóricas 
        incluidas en columns que cumplan que su test de relación con "target_col" es significatio para el nivel 1-pvalue de significación 
        estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 

        Argumentos:
        df (DataFrame): Dataframe a analizar.
        target_col (DataFrame): Columna objetivo para calcular las correlaciones. Valor por defecto = "".
        columns (list): features a relacionar con la columna "target_col". Valor por defecto = [].
        pvalue (float): Opcional, el valor límite del p-valor para la prueba t de una muestra. Número decimal entre 0 y 1. Valor por defecto = 0.05.
        with_individual_plot (bool): Indica si se agrupan los histogramas en uno solo. Valor por defecto = False

        Retorna:
        tipo (list): los valores de "columns" que cumplan con las condiciones anteriores. 
    """

    # 1. Validación de entrada
    if target_col == "" or target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        return None
    
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es numérica.")
        return None
    
    if not (isinstance(pvalue, float) or isinstance(pvalue, int)):
        print("Error: El valor de 'pvalue' no es adecuado. Debe ser un float o entero.")
        return None
    if not (0 <= pvalue <= 1):
        print("Error: El valor de 'pvalue' no es adecuado. Debe ser estar entre 0 y 1.")
        return None
    if not target_col in df.columns:
        print(f"El {target_col} no se encuentra dentro del DataFrame.")
        return None
    for col in columns:
        if not col in df.columns:
            print(f"La columna {col} no se encuentra dentro del DataFrame.")
            return None

    # 2. Validación de cardinalidad
    calculo_cardi_target = df[target_col].nunique()
    porcentaje_cardi_target = df[target_col].nunique()/len(target_col) * 100
    #Comporbación de cardinalidad
    if not (calculo_cardi_target >= 10) | (porcentaje_cardi_target >= 30):
        print(f"La columna {target_col} no es una variable numerica continua o discreta ")
        return None

    # 3. Comprobación columnas DataFrame son categoricas y numericas (solo usariamos las numericas en caso de que la lista columns == [])
    columnas = df.columns
    lista_categoricas = []
    lista_numericas = []
    for col in columnas:
        calculo_cardi = df[col].nunique()
        if calculo_cardi < 10:
            lista_categoricas.append(col)
            continue
        else:
            lista_numericas.append(col)
            continue
    
    # 3.1 Si no hay columnas categoricas
    # Si la lista 'lista_categoricas' está vacía, seleccionamos las variables numéricas del DataFrame
    if len(lista_categoricas) == 0:
        print("No hay columnas categóricas en el DataFrame.")
        lista_categoricas = df.select_dtypes(include=[np.number]).columns.tolist()
        lista_categoricas.remove(target_col)  # Excluimos la columna target


    #4. Tratamos Nan en el DataFrame y añadimos las features
    features = []
    
    # #4.1 SI LISTA COLUMNS TIENE DATOS
    if columns != []:
         for col in columns: #columns iniciales
             # Eliminamos filas con valores NaN en el DataFrame
             df_sin_nan = df.dropna()
             parametro = len(df_sin_nan[col].unique())
             if parametro <= 1:
                 # Columna categórica sin suficiente variación
                 continue
             if parametro == 2:
                 # Test t de Student si es Binaria
                 grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
                 stat, p = stats.ttest_ind(*grupo_cat, equal_var=False)
             else:
                 # ANOVA si hay más de dos categorías
                 grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
                 stat, p = stats.f_oneway(*grupo_cat)
                 warnings.filterwarnings("ignore", category= ConstantInputWarning) #Quitar aviso de baja variacion, ya lo revisamos en parametro ==1 y == 2
             if p < pvalue:
                 features.append(col)

    #4.2 SI LISTA COLUMNS ESTA VACIA
    if columns == []:
        for col in lista_numericas: #columns iniciales
            # Eliminamos filas con valores NaN en el DataFrame
            df_sin_nan = df.dropna()
            parametro = len(df_sin_nan[col].unique())
            if parametro <= 1:
                # Columna categórica sin suficiente variación
                continue
            if parametro == 2:
                # Test t de Student si es Binaria
                grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
                stat, p = stats.ttest_ind(*grupo_cat, equal_var=False)
            else:
                # ANOVA si hay más de dos categorías
                grupo_cat = [df_sin_nan.loc[df_sin_nan[col] == cat][target_col] for cat in df_sin_nan[col].unique()]
                stat, p = stats.f_oneway(*grupo_cat)
                warnings.filterwarnings("ignore", category= ConstantInputWarning) #Quitar aviso de baja variacion, ya lo revisamos en parametro ==1 y == 2
            if p < pvalue:
                features.append(col)

    #5. Pintar histogramas de las variables
    # Si with_individual_plot es True, generamos el histograma agrupado
    if with_individual_plot:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df_sin_nan[features], x=target_col, hue=col, multiple="stack", palette="tab10")
        plt.title(f"Histograma agrupado para {target_col} por {col} (p-value: {p:.4f})")
        plt.xlabel("Fare")
        plt.ylabel(target_col)
        plt.legend(title=col)
        plt.show()
    else:
        for feature in features:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df_sin_nan[feature], palette="tab10")
            plt.title(f"Histograma simple para {target_col} por {feature} (p-value: {p:.4f})")
            plt.xlabel(feature)
            plt.ylabel(target_col)
            plt.legend(title=col)
            plt.show()
    # Retornamos las columnas que han pasado el test de significancia
    return features if features else None

