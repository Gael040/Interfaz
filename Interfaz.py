import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Alias, Ord, Card
from copy import deepcopy

st.title('CLAIO Interfaz, La buena :]')


# Cargar DataFrame con las distancias entre cada lugar
datos = pd.read_csv('distancias.csv')

# Se crea una función para transformar el formato del dataframe
@st.cache_data
def transformar_df(df):
    lista_dist = []

    for TO in range(1, df.shape[1]):

        for FROM in range(1, df.shape[1]):
            lista_dist.append([str(TO), str(FROM), df.iloc[TO-1, FROM]])

    df2 = pd. DataFrame(lista_dist, columns=["from", "to", "distance"]).set_index(["from", "to"])
    return df2

tabla = transformar_df(datos)


@st.cache_data
def transformar_pesos(act):
    pesos = pd.read_csv('pesos3.csv')
    peso = 3
    pesos.loc[pesos['Tipo'] == act, 'Pesos'] = peso
    pesos = pesos[['Nodo','Pesos']]

    weights = []

    for row in range(pesos.shape[0]):
      nodo = pesos['Nodo'][row]
      nodo = str(nodo)
      peso = pesos['Pesos'][row]
      weights.append([nodo, peso])
    print(weights)
    return weights

transformar_pesos(1) # SE le debe de preguntar al usuario.


@st.cache_data
def calcula_costo(todos_costos,costos_lugares_no_fue):
    print(costos_lugares_no_fue)
    print(todos_costos)
    suma_total = 0
    for lug in range(len(todos_costos)):
        suma_total = todos_costos[lug][1] + suma_total
    print('Costo total:', suma_total)
    suma_parcial = 0
    for lug in range(len(costos_lugares_no_fue)):
        suma_parcial = costos_lugares_no_fue[lug][1] + suma_parcial
    print('Costo que no fue :', suma_parcial)

    costo_acumulado=suma_total-suma_parcial

    print('Costo fue :', costo_acumulado)

    return costo_acumulado


places= pd.read_csv("pesos3.csv")

lugares_mas_nodos=places.iloc[:,[0,1]]

@st.cache_data
def nodos_a_lugares(camino):
    contador=0
    for i in camino:
        #print(i)  
        origen, destino = i  

        for _, row in lugares_mas_nodos.iterrows():
            if origen == row['Nodo']:
                #print(origen)
                st.text(row[0])  
                break 

        for _, row in lugares_mas_nodos.iterrows():
            if destino == row['Nodo']:
                #print(destino)
                #print(row[0])  
                break  
        contador+=1
    if contador == len(camino):
        st.text(lugares_mas_nodos.iloc[0,0])
        


lugares = list(range(1, datos.shape[0]+1))



def ejecutar_proceso(tabla,tiempLug,tiempo_max,costosLug,weights,presupuesto_max,numero_de_dias):

    # lugares = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    costo_acumulado=0
    todos_costos= costosLug.copy()
    costos_lugares_no_fue= costosLug.copy()

    for _ in range(numero_de_dias):

      m = Container()

      # Sets
      #lugares = [1,2,3,4,5,6]
      i = Set(container=m, name="i", description="punto i", records=lugares)
      j = Alias(m, name = "j", alias_with = i)
      k = Alias(m, name = "k", alias_with = i)
      #print(i.records)

      # Variables
      x = Variable(
          container=m,
          name="x",
          domain=[i, j],
          type="Binary",
          description="Si va del punto i al j",
      )
      y = Variable(
          container=m,
          name="y",
          domain=[i],
          type="Binary",
          description="Si pasa por el punto i",
      )
      u = Variable(
          container=m,
          name="u",
          domain=[i],
          type="Positive",
          description="Orden",
      )

      # Parámetros

      d = Parameter(
          container=m,
          name="d",
          domain=[i],
          description="parametro de tiempo en lugar",
          records=tiempLug,
      )

      p = Parameter(
          container=m,
          name="p",
          domain=[i],
          description="parametro de costos en el lugar",
          records=costosLug,
      )

      w = Parameter(
          container=m,
          name="w",
          domain=[i],
          description="Beneficio de visitar",
          records=weights,
      )

      tiempos = Parameter(
          container=m,
          name="cT",
          domain=[i, j],
          description="tiempo que tarda en ir de i a j",
      )
      tiempos.setRecords(tabla.reset_index())
      #print(tiempos.records)

      # Ecuaciones
      objective = Sum((i,j), w[i]*x[i,j])

      # r1(i)$(ord(i) = 1).. sum(j$(ord(j)>1), x(i,j)) =e= 1;
      r1 = Equation(
          container=m,
          name="r1",
          domain=[i],
          description="R Nodo Origen",
      )
      r1[i].where[Ord(i)==1]=Sum(j.where[Ord(j)>1], x[i, j]) == 1

      # r2(j)$(ord(j) = 1).. sum(i$(ord(i)>1), x(i,j)) =e= 1;
      r2 = Equation(
          container=m,
          name="r2",
          domain=[j],
          description="R Nodo Destino",
      )
      r2[j].where[Ord(j)==1]=Sum(i.where[Ord(i)>1], x[i, j]) == 1

      # r3(i).. sum(j,x(i,j))-sum(k,x(k,i))=e=0;
      r3 = Equation(
          container=m,
          name="r3",
          domain=[i],
          description="R Nodos intermedios",
      )
      r3[i]=Sum(j, x[i, j])-Sum(k, x[k, i]) == 0

      # r5(i)$(ord(i) > 1).. 2=l=u(i);
      r5 = Equation(
          container=m,
          name="r5",
          domain=[i],
          description="R Orden menor",
      )
      r5[i].where[Ord(i)>1]= u[i] >= 2

      # r6(i)$(ord(i) > 1).. u(i)=l=card(i);
      r6 = Equation(
          container=m,
          name="r6",
          domain=[i],
          description="R Orden mayor",
      )
      r6[i].where[Ord(i)>1]= u[i] <= Card(i)

      # r7(i,j)$(ord(i) > 1 and ord(j) > 1).. u(i)-u(j)+1=l=(card(i)-1)*(1-x(i,j));
      r7 = Equation(
          container=m,
          name="r7",
          domain=[i,j],
          description="R Ciclos",
      )
      r7[i,j].where[Ord(i)>1 and Ord(j)>1]= u[i]-u[j]+1 <= (Card(i)-1)*(1-x[i,j])

      # r8.. sum(i,sum(j,c(i,j)*x(i,j))) + sum(i,d(i)*y(i)) =l= 439;
      r8 = Equation(
          container=m,
          name="r8",
          description="R Tiempo",
      )
      r8[i,j]= Sum(i,Sum(j,tiempos[i,j]*x[i,j])) + Sum(i,d[i]*y[i]) <= tiempo_max

      # r9(i).. y(i) =g= sum(j, x(i,j));
      r9 = Equation(
          container=m,
          name="r9",
          domain=[i],
          description="R Visitar lugares",
      )
      r9[i]= y[i] >= Sum(j,x[i,j])

      #Restricción presupuesto

      r10 = Equation(
          container=m,
          name="r10",
          description="R Tiempo",
      )
      r10[i]= Sum(i,p[i]*y[i]) <= presupuesto_max -costo_acumulado



      # Se inicializa el modelo y resuelve
      elizalde = Model(
          container=m,
          name="elizalde",
          equations=m.getEquations(),
          problem="MIP",
          sense=Sense.MAX,
          objective=objective,
      )

      # Lo comentado sirve para visualizar información del modelo
      import sys
      elizalde.solve(output=sys.stdout)
      elizalde.solve()
      elizalde.objective_value

      if str(elizalde.status) == 'ModelStatus.IntegerInfeasible':
        st.text('INFEASIBLE')
        break

      # Se crea un DataFrame con las respuestas
      tax=x.records.set_index(["i", "j"])
      #print(tax['level'].items())
      nodos_interes = []

# Ciclo para encontrar los nodos con valor 1
      for indice, valor in tax['level'].items():
        if valor == 1:
            print("Nodos:")
            print(indice)
            nodos_interes.append((int(indice[0]), int(indice[1])))  # Guardar los pares (i, j)

# Ordenar los nodos según el camino que inicia y termina en 1
      conexiones = {inicio: fin for inicio, fin in nodos_interes}
      print("conexiones", conexiones)
      camino = []
      nodo_actual = 1

      while True:
        siguiente = conexiones.get(nodo_actual)
        if not siguiente:
            break  # Salimos si no hay más conexiones
        camino.append((nodo_actual, siguiente))
        nodo_actual = siguiente
        if nodo_actual == 1:
            break  # Si volvemos a 1, cerramos el ciclo

    # Imprimir el camino ordenado
      print("Camino ordenado:", camino)
      
      #st.text(camino)
      nodos_a_lugares(camino)


      # Ciclo for para borrar en el DataFrame tabla y en tiempLug los nodos visitados
      for indice in camino:
        print(indice[1])
        numero_a_borrar = indice[1]
        numero_a_borrar2 = indice[0]

        if numero_a_borrar == 1:
            print("No lo borra")
        else:
            for i in lugares:
                tabla = tabla.drop((str(i), str(numero_a_borrar)))

            tabla = tabla.drop(index=str(numero_a_borrar))
            lugares.remove(numero_a_borrar)

            for xet in tiempLug:
                if xet[0] == str(numero_a_borrar):
                    valor3 = xet[1]

            tiempLug.remove([str(numero_a_borrar), valor3])

            for xet2 in costosLug:
                if xet2[0] == str(numero_a_borrar):
                    costo_lugar_visitado = xet2[1]


            costosLug.remove([str(numero_a_borrar), costo_lugar_visitado])

            for xetW in weights:
              if (xetW[0]== str(numero_a_borrar)):
                valor4= xetW[1]

            weights.remove([str(numero_a_borrar),valor4])



            costos_lugares_no_fue=costosLug


            costo_dia=calcula_costo(todos_costos,costos_lugares_no_fue)


            todos_costos=costos_lugares_no_fue.copy()


            costo_acumulado= costo_acumulado+ costo_dia


    return elizalde

# Lista con los tiempos de visita
tiempLug = [['1',0],['2',60],['3',60],['4',120],['5',120],['6',120],['7',120],['8',120],['9',120],['10',60],['11',60],['12',45],['13',90],['14',300],['15',120],['16',90],['17',60],['18',120],['19',60],['20',60]]
costosLug = [['1',0],['2',60],['3',60],['4',120],['5',120],['6',120],['7',120],['8',120],['9',120],['10',60],['11',60],['12',45],['13',90],['14',300],['15',120],['16',90],['17',60],['18',120],['19',60],['20',60]]

st.subheader("Elige tu tipo de actividad favorita")

st.markdown("1- Cenote")
st.markdown("2- Playas")
st.markdown("3- Parques")
st.markdown("4- Culturales")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:10px;
}
</style>
''', unsafe_allow_html=True)

act=st.number_input("1-4",min_value=1, max_value=4, step=1)
act = int(act)

weights = transformar_pesos(act)

tiempo_dia=st.number_input(
    '''Elige el tiempo máximo que puedes destinar para ser turista cada día
''',min_value=1,step=1)
tiempo_max = int(tiempo_dia)



presupuesto_max=st.number_input(
    '''Elige el presupuesto máximo para tus vacaciones
''',min_value=1,step=1)
presupuesto_max = int(presupuesto_max)

numero_de_dias = st.number_input(
    "Elige la cantidad de días de tus vacaciones",min_value=1,step=1)
numero_de_dias = int(numero_de_dias)

#page_bg_img = f"""
#<style>
#[data-testid="stAppViewContainer"] > .main {{
#background-image: url("https://htmlcolorcodes.com/assets/images/colors/pastel-blue-color-solid-background-1920x1080.png");
#background-size: cover;
#background-position: center center;
#background-repeat: no-repeat;
#background-attachment: local;
#}}
#[data-testid="stHeader"] {{
#background: rgba(0,0,0,0);
#}}
#</style>
#"""
#st.markdown(page_bg_img, unsafe_allow_html=True)



tabla_modificada = ejecutar_proceso(tabla,tiempLug,tiempo_max,costosLug,weights,presupuesto_max,numero_de_dias)


st.subheader('Mapa de rutas turisticas')
st.map()