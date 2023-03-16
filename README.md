# Lab6_IA
# Exploración de datos (Task 1)
## Encoding
No fue necessario codificar ninguna variable categorica puesto que las variables elegidas ya venian codificadas o eran cuantitativas.  
## Balanceo de dataset
### Lol
Se hizo la revisión del balance de los datos utilizando la funcion nativas de pandas valuecount(). De esto se obtuvo que se tienen. 
- 4949 valores 0 (False). 
- 4930 valores 1 (True).
Se considera balanceado por lo que no se realizo ninguna acción al respecto. 
### Fifa
Dado que para FIFA la variable objetivo era cuantiativa se obvio el balanceo. 
## Escalamiento de variables
No es necesario dado el modelo a implementar
## Seleccion de variables
### Lol
Las variables seleccionadas se realizaron considerando la relación, de acuerdo al impacto en partida de acuerdo a la experiencia propia, así como la eliminación de variables que de una otra forma estaba representadas por otra. Se consideró por ejemplo que las kils de los azules son prácticamente las deaths de los rojos. 
 
### Variables seleccionadas
- blueWins -> Variable objetivo. 
- blueKills -> Implica la ventaja en habilidad que se tenga de un equipo en relación a otro en algún momento durante la partida. 
- blueDeaths -> Compitiendo con la variable anterior, representa l habilidad de los rojos sobre los azules en momentos de la partida. 
- blueTowersDestroyed -> Representa la vulnerabilidad que existieron en los carriles de los azules. También es natural pensar que entre más torres destruidas más fácil es que hayan perdidos, pues existen más formas y estrategias para llegar a destruir el nexo por el equipo contrario.
- redTowersDestroyed  -> Lo mismo que la anterior pero en favor de los azules. 
- blueTotalGold -> El oro es muy importante, representa su capacidad de comprar items que mejora el pvp y mejora la situación de un equipo en general.
- redTotalGold -> Lo mismo que la anterior pero en favor del equipo rojo. 

### Fifa

Las variables seleccionadas inicialmente fueron las siguientes, se baso en lo que relata el mismo kaggle sobre las variables y la experiencia de conocidos en el juego. Finalmente en el modelo, se eliminaron algunas de variables (viendo la corrleación de ellas) más que nada para mejorar el desempeño de la contruaccíon del árbol. 

- Age
- Overall
- Value
- Wage
- Special
- Acceleration
- Dribbling
- Finishing
- Agility
- Ball control
- Composure
- Potential
## Métrica de desempeño
### Fifa
Dado que la predicción se hara con un valor aproximado (la media) del sector en el que seleccione el árbol, la mejor métrica es la **media de R2** que finalmente dira que tanto se alejó la predicción realiazada de la realidad.

### Lol 
Dado que la predicción se hara con valores 1 o 0, booleano la métrica de desempeño se hará con acuraccy calculando cuantas veces acertó exactamente al valor esperado. 

## Comparativa 

### League of legends
**¿Qué implementación fue mejor? ¿Por qué?**
Ambas dieron una métrica de desempeño prácticamente igual y creo que fue porque la decisión sobre un valor booleano, realizado bien el DTC es más probable de acertar. 


### FIFA
**¿Qué implementación fue mejor? ¿Por qué?**
El r2 de sklearn fue mucho menor. En este caso creemos fue así porque los splits del árbol existía una variedad de casos por feature. Y, finalmente,el valor de la media tiende a hacer un tanto impreciso el valor predicho según el set de datos, intuyo que la técnica de la predicción de sklearn es un tanto distinta.  