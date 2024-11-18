import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import f_oneway, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

def describe_df(df):
    '''
    Esta función muestra información específica del DF original. 
    Esa información será: el tipo de objeto, el % de valores nulos o missings,
    los valores únicos y el % de cardinalidad de cada columna del DF original.

    Argumentos:
    df (pd.DataFrame): DF original sobre el que queremos recibir la información.

    Retorna:
    pd.DataFrame: Df con la información específica.
    '''

    #Creo un diccionario con la columna que va a estar fija
    #Y después añadir las columnas del DF original

    dict_col = {'COL_N': ['DATA_TYPE', 'MISSINGS (%)', 'UNIQUE_VALUES', 'CARDIN (%)']}

    #Fórmula para calcular el porcentaje de nulos
    na_ratio = ((df.isnull().sum() / len(df))*100)

    #Añado al diccionario como clave el nombre de las columnas, y como valores
    #la información del describe
    for col in df:
        dict_col[col] = [df.dtypes[col], na_ratio[col], len(df[col].unique()), round(df[col].nunique()/len(df)*100,2)]

    # Creo el DF.describe
    df_describe = pd.DataFrame(dict_col)

    return df_describe



def tipifica_variables(df, umbral_categoria, umbral_continua):
    '''
    Esta función sirve para poder tipificar las variables de un DF dado en categórica, numerica continua o numerica discreta.

    Argumentos:
    df (pd.DataFrame): DF original para adquirir las variables que se quiera tificar.
    umbral_categoria (int): un entero donde corresponda al umbral que queramos asignar a una variable categórica.
    umbral_continua (float): un float donde corresponda al umbral que queramos asignar a una variable numérica.

    Retorna:
    pd.DataFrame: Un DF con dos columnas 'nombre_varibale' y 'tipo_sugerido', que tendrá tantas filas como columnas haya en el DF original.
    '''

    if type(umbral_categoria) != int:
        raise TypeError(f'El valor de "umbral_categoria" debe ser de tipo {int}, pero recibió de tipo {type(umbral_categoria)}')

    elif type(umbral_continua) != float:
        raise TypeError(f'El valor de "umbral_continua" debe ser de tipo {float}, pero recibió de tipo {type(umbral_continua)}')

    else:

        df_tipificacion = pd.DataFrame({
            'nombre_variable': df.columns
        })

        df_tipificacion['tipo_sugerido'] = ''

        for i, val in df_tipificacion['nombre_variable'].items():
            card = df[val].nunique()
            porcentaje = df[val].nunique()/len(df) * 100

            if card == 2:
                df_tipificacion.at[i,'tipo_sugerido'] = 'Binaria'
            
            elif card < umbral_categoria:
                df_tipificacion.at[i,'tipo_sugerido'] = 'Categórica'
        
            else:
                if porcentaje > umbral_continua:
                    df_tipificacion.at[i, 'tipo_sugerido'] = 'Numérica Continua'
                else:
                    df_tipificacion.at[i, 'tipo_sugerido'] = 'Numérica Discreta'
    
    return df_tipificacion


def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):
    '''
    Obtiene las columnas numéricas cuya correlación con target_col es significativa.
    
    Argumentos:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo (debe ser numérica).
    umbral_corr (float): Umbral de correlación para filtrar las columnas.
    pvalue (float, optional): Umbral de significancia para el valor p. Si es None, solo se considera el umbral de correlación.
    
    Returns:
    List[str]: Lista de nombres de columnas que cumplen con los criterios.
    '''
    if type(umbral_corr) != float: # Chequeo si el umbral_corr es float
        print(f'El valor dado en "umbral_corr" debe ser de tipo {float}, pero recibió un valor de tipo {type(umbral_corr)}')
        return None
    if umbral_corr < 0 or umbral_corr > 1: # Chequeo si el umbral_corr está entre 0 y 1
        print(f'El valor de "umbral_corr" no está entre 0 y 1 ya que el valor es {umbral_corr}')
        return None
    if not is_numeric_dtype(df[target_col]): # Chequeo si es numérica 
        print(f'La target "{target_col}", no es de tipo numérico, ya que esta variable es de tipo {df[target_col].dtypes}')
        return None
    if df[target_col].nunique() < 10:
        print(f'Error: La columna target "{target_col}" no tiene alta cardinalidad, ya que tiene menos de 10 valores únicos.')
        return None
    
    else: # Si esta todo correcto
        corr = np.abs(df.corr(numeric_only=True)[target_col]).sort_values(ascending=False) # correlaciones de todas las numericas con la target en absoluto
        corr_list = [i for i,val in corr.items() if val > umbral_corr] # lista de las variables que superen el umbral_corr
        corr_list.remove(target_col) # elimino la target de la lista

        if pvalue == None: # además, si pvalue es None
            print(f'Las correlaciones con las demás variables numéricas y "{target_col}" son:')
            print()
            print(corr)
            print('\nLista de variables numéricas que tiene correlación según el umbral dado en la función:')
            return corr_list # solo devuelvo las correlaciones y las variables
        
        else: # si ponen un valor en pvalue
            if type(pvalue) != float:
                raise TypeError(f'El valor de "pvalue" debe ser de tipo {float}, pero recibió de tipo {type(pvalue)}')
            if pvalue <= 0 or pvalue >= 1:
                raise ValueError(f'El valor de "pvalue" debe estar entre 0 y 1, pero el valor dado fue {pvalue}')
            else:
                pv_list = []
                for val in corr_list: #itera sobre cada una de las variables que SÍ superan el umbral_corr
                    _, p_value = pearsonr(df[val], df[target_col]) #Test de pearsonr
                    if p_value < (1-pvalue): # Condición 
                        pv_list.append(val)
                        
                print(f'Las correlaciones con las demás variables numéricas y "{target_col}" son:')
                print()
                print(corr)
                print('\nLista de variables numéricas que tiene correlación según el umbral dado en la función:')
                print(corr_list)
                print(f'\nLas features numéricas con una significancia del {pvalue* 100} % son:')
                return pv_list # Muestro además las variables con significancia



def plot_features_num_regression (df, target_col="", columns=[], umbral_corr=0, pvalue=None):

    """

    Esta función sirve para visualizar aquellas variables que tengan correlación con la variable objetivo.

    Argumentos:
    df (pd.DataFrame): DataFrame que contiene los valores.
    target_col (str): Teniendo un valor por defecto "".
    columns(List[str]): Lista de strings cuyo valor por defecto es una lista vacía.
    umbral_corr(float): Con valor por defecto 0
    pvalue(float): Umbral de significancia para el valor p. Por defecto, con valor None

    Retorna:
    Pairplot del DataFrame consirando los valores adjuducados en cada argumento de la función.

    """
    valid_columns = [] # Pongo fuera la lista vacia para que valga tanto para si hay lista vacia como si no

    if type(umbral_corr) != float: # Chequeo si el umbral_corr es float
        print(f'El valor dado en "umbral_corr" debe ser de tipo {float}, pero recibió un valor de tipo {type(umbral_corr)}')
        return None
    if umbral_corr < 0 or umbral_corr > 1: # Chequeo si el umbral_corr está entre 0 y 1
        print(f'El valor de "umbral_corr" no está entre 0 y 1 ya que el valor es {umbral_corr}')
        return None
    if not is_numeric_dtype(df[target_col]): # Chequeo si es numérica 
        print(f'La target "{target_col}", no es de tipo numérico, ya que esta variable es de tipo {df[target_col].dtypes}')
        return None
    if df[target_col].nunique() < 10: # Chequeo que target col tenga por lo menos 10 valores diferentes para asegurarme la cardinalidad.
        print(f'Error: La columna target "{target_col}" no tiene alta cardinalidad, ya que tiene menos de 10 valores únicos.')
        return None
    if pvalue is not None: # Chequeo que el pvalue esté entre 0 y 1.
        if not isinstance(pvalue, float):
            raise TypeError(f'El valor de "pvalue" debe ser de tipo {float}, pero recibió de tipo {type(pvalue)}')
        if pvalue <= 0 or pvalue >= 1:
            raise ValueError(f'El valor de "pvalue" debe estar entre 0 y 1, pero el valor dado fue {pvalue}')    

    else: # Si esta todo correcto
        if not columns: # Comprobamos que la lista no esté vacía. 
            # Si la lista de columnas está vacía, la igualamos a las columnas numéricas del dataframe
            columns = df.select_dtypes(include='number').columns.tolist()
            columns.remove(target_col)
            print("Como no hay columnas en la lista del argumento usamos las numéricas, que son: ", columns)
        # Filtrar columnas que cumplen con la correlación y el p-value
            for col in columns:
                if col != target_col:
                    corr, p_val = pearsonr(df[target_col], df[col])
                    if abs(corr) > umbral_corr and (pvalue is None or p_val < (1 - pvalue)):
                        valid_columns.append(col)
            print("Las columnas válidas para el pairplot son", valid_columns)
                
    #if valid_columns:
            sns.pairplot(df, vars=valid_columns, hue=target_col)
            plt.show()
            return valid_columns
        
        else: # Si hay una lista de variables
            for col in columns:
                if col != target_col:
                    corr, p_val = pearsonr(df[target_col], df[col])
                    if abs(corr) > umbral_corr and (pvalue is None or p_val < (1 - pvalue)):
                        valid_columns.append(col)
            print("Las columnas válidas para el pairplot son", valid_columns)
            
            for i in range(0, len(valid_columns), 4):
                subset = valid_columns[i:i+4]
                sns.pairplot(df, vars=[target_col] + subset, hue=target_col)
                plt.show()
                return valid_columns

          

def get_features_cat_regression (df, target_col, pvalue = 0.05):
    ''' Esta funcion devueleve una lista con las variables categoricas del dataframe que guardan una relacion
        siginificativa con la variable target, superando el test ANOVA con una confianza estadistica del 95%. 
        Evalua los argumentos de entrada y retornara None en caso de que alguno de los valores de entrada no sean 
        adecuados. 

        Argumentos:
        df : (pd.Dataframe) Dataframe con las variables que se quieren testar.
        target_col : (df["columna_target"]) Columna del dataframe que se toma como objetivo (y).
        pvalue : (float) Por defecto 0.05. Umbral de confianza estadistica. 

        Retorna:
        Lista con las variables categoricas del dataframe. 
    '''

    # Verificar que los argumentos sean validos:

    if not isinstance(df, pd.DataFrame): # Verifica que el argumento 'df' sea efectivamente un dataframe.
        print("El argumento 'df' introducido no es válido")
        return None
    
    if target_col not in df.columns: #Verifica que la columna target que hemos pasado como argumento se encuentre en el dataframe.
        print("El argumento 'target_col' introducido no se encuentra en el dataframe")
        return None
    
    if target_col not in df.select_dtypes(include=[int, float]).columns or df[target_col].nunique() < 10:
        # Comprueba que la columna target sea numérica (int o float) continua:
        print("El argumento 'target_col' no es una variable numérica continua o discreta con alta cardinalidad.")
        return None
    
    if not isinstance(pvalue, float):
        raise TypeError(f'El valor de "pvalue" debe ser de tipo {float}, pero recibió de tipo {type(pvalue)}')
    if pvalue <= 0 or pvalue >= 1:
        raise ValueError(f'El valor de "pvalue" debe estar entre 0 y 1, pero el valor dado fue {pvalue}')

    
    # Abrimos una lista para almacenar las columnas clasificadas como categóricas que superen el test:

    categoricas = []

    for columna in df.columns:   

        if columna != target_col: # Salta la columna target

            if df[columna].dtype == object or df[columna].nunique() < 10 : # Buscamos las columnas categoricas que haya en el df, si la encuentra realiza el test:
            
                grupos = df[columna].unique()  # Obtiene los valores únicos de la columna categórica
                categoria = [df[df[columna] == grupo][target_col] for grupo in grupos] # Filtra la columna del df por categoria y valores de la columna target
    
                _, p_val = f_oneway(*categoria) # Test ANOVA
                
                if p_val < pvalue: # Si el resultado del pvalor del test es menor que nuestro umbral, lo metemos en la lista
                    categoricas.append(columna)
                    print(f'{columna} --> {p_val}')
    
    return print(f"Las variables categóricas encontradas son: {categoricas}")



def plot_features_cat_regression (df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Esta función realiza un análisis estadístico entre la columna 'target_col' y las variables categóricas 
    en 'columns', pintando histogramas agrupados solo para aquellas que superan el nivel de significación estadística 
    dado por el valor de 'pvalue'. Si 'columns' está vacío, se toman las variables numéricas del dataframe.

    Argumentos:
    df (pd.DataFrame): El dataframe que contiene los datos.
    target_col (str): Columna objetivo, debe ser numérica contínua o discreta con alta cardinalidad. Valor por defecto es una cadena vacía.
    columns (list): Lista de variables categóricas a evaluar. Valor por defecto es una lista vacía.
    pvalue (float): Nivel de significancia estadística, por defecto 0.05.
    with_individual_plot (bool): Si es True, generará un histograma por cada variable categórica individualmente. Valor por defecto False.

    Retorna:
    List[str]: Lista de columnas categóricas que tienen una relación estadísticamente significativa con la variable objetivo.
    """

    # Verificar que df sea un DataFrame válido
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' no es un DataFrame válido.")
        return None

    # Verificar que target_col sea una variable válida en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no se encuentra en el DataFrame.")
        return None

    # Verificar que pvalue sea un float válido entre 0 y 1
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print(f"Error: El valor de 'pvalue' debe ser un float entre 0 y 1, pero recibió {pvalue}.")
        return None

    # Si la lista 'columns' está vacía, asignar todas las columnas numéricas del DataFrame excepto target_col
    if not columns:
        columns = [col for col in df.select_dtypes(include=[int, float]).columns if col != target_col]
    
    # Verificar que la columna target sea numérica contínua o discreta con alta cardinalidad
    if target_col not in df.select_dtypes(include=[int, float]).columns or df[target_col].nunique() < 10:
        print("El argumento 'target_col' no es una variable numérica continua o discreta con alta cardinalidad.")
        return None

    # Lista para almacenar las variables que pasan el test de significancia
    significant_columns = []

    # Recorremos todas las columnas en la lista 'columns'
    for col in columns:
        
        if col in df.select_dtypes(include=['object']).columns or df[col].nunique() < 10:  # Variables categóricas
            # Realizamos el test ANOVA entre la variable continua 'target_col' y la categórica 'col'
            grupos = df[col].unique()
            grupos_datos = [df[df[col] == grupo][target_col] for grupo in grupos]
            _, p_val = f_oneway(*grupos_datos)

            # Si la relación es significativa, agregamos la columna y pintamos los gráficos
            if p_val < pvalue:
                significant_columns.append(col)

                # Graficar histogramas agrupados por categorías
                if with_individual_plot:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x=target_col, hue=col, multiple="stack")
                    plt.title(f'Histograma de {target_col} agrupado por {col}')
                    plt.show()

        else:  # Variables numéricas
            # Calcular la correlación de Pearson entre target_col y cada variable numérica
            corr, p_val = pearsonr(df[target_col], df[col])

            # Verificar si la correlación y el p-valor cumplen con los umbrales
            if p_val < pvalue:
                significant_columns.append(col)

                # Graficar la relación si se solicita
                if with_individual_plot:
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x=df[col], y=df[target_col], line_kws={"color": "red"})
                    plt.title(f'Relación entre {target_col} y {col} (Correlación: {corr:.2f}, p-value: {p_val:.4f})')
                    plt.xlabel(col)
                    plt.ylabel(target_col)
                    plt.show()

    # Si no hay columnas significativas, mostrar un mensaje
    if not significant_columns:
        print("No se encontraron columnas con una relación estadísticamente significativa con la variable objetivo.")

    return significant_columns

