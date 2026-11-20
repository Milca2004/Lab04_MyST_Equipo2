# Solicitudes de Arturo a Milca (revisión de `milca/datos`)

Salen de mi auditoría (código corrido el 2026-11-19) y del council (2). Les quité al council lo que no sobrevivió a la verificación:

- "Rellenos tras huecos": `data.py` no rellena precios.
- "Falta una variable de hueco": ya va en mi observación como `log1p(gap_hours)`.
- "Los regímenes contaminan el fold 1": solo se usan en el EDA, no en el estado ni en ninguna decisión.

Ninguna de estas solicitudes bloquea a Paula para entrenar. Las BLOQUEANTES lo son para la **entrega**: sin ellas faltan secciones o artefactos exigidos.

## BLOQUEANTE para la entrega

1. **Cargador del S&P 500 (sección 3.7).**
   - **Archivo:** `code/data.py`, por ejemplo `load_sp500()`, más el archivo de datos.
   - **Cambio:** decidir de dónde sale el archivo y registrarlo en el README.
   - **Alineación:** el S&P no opera fines de semana ni de noche, así que los retornos se comparan en **diario**. Toma el cierre diario de BTC a la hora del cierre de NYSE (o 00:00 UTC, declarado) y aplica 0.125% al comprar y al vender el índice.
   - **Razón:** el enunciado exige este benchmark y a la vez dice "datos externos de precio: no permitidos". Hay que mandar hoy un correo al profe preguntando si el S&P es la excepción y guardar la respuesta. Si no responde, usar buy & hold de BTC como benchmark principal y declarar la limitación.
2. **Secciones de reporte y walkthrough de las preguntas 1 y 2.**
   - **Archivo:** `docs/borradores/milca_*.md` o el equivalente.
   - **Cambio:** pasar a texto las cifras que ya calcula `characterize_regimes` (7 regímenes con retorno, volatilidad anualizada, drawdown y curtosis) y la tabla ADF/KPSS. Nombrar la alternativa rechazada (variables en nivel).
   - **Razón:** `docs/` está vacío y estas dos preguntas valen crédito en C y en la exposición.
3. **Merge de `milca/datos` a `main`.**
   - **Cambio:** abrir el PR (yo lo reviso, como acordamos).
   - **Razón:** `main` sigue en el esqueleto y se califica el último commit de `main`. Mi PR de `arturo/entorno` contiene tus commits, así que fusionar el tuyo primero deja el mío con un diff limpio.

## IMPORTANTE (calidad de la justificación de la pregunta 2)

4. **Variables no estacionarias dentro del estado.**
   - **Hallazgo:** en el train del fold 1, ADF+KPSS marcan como NO estacionarias `vol_24`, `vol_168`, `range_1`, `parkinson_24` y `trend_std` (`feature_analysis_fold_1.csv`, columna `estacionaria`), y las cinco están en el estado.
   - **Opción A (preferida):** sustituirlas por razones adimensionales, como log(vol_24/vol_168), range_1/vol_24 o trend_mean/trend_std, y repetir ADF/KPSS en el train de los **4** folds.
   - **Opción B:** dejarlas y argumentar que son información de **costo**, no de dirección. Las barreras miden ±vol·√H, así que la volatilidad decide si la ida y vuelta de 0.25% cabe en el movimiento. Su información mutua (0.12–0.13 contra un nulo ≈0) lo respalda.
   - **Aviso:** cualquiera de las dos cambia la observación. Avísanos antes, porque Paula entrena sobre ella.
5. **Rechazo del nivel de precio.**
   - **Hallazgo:** `close` y `log_close` tienen MI de 0.15–0.18 a h24, más que `dist_ema_168` (0.058).
   - **Cambio:** apoyar el rechazo en que no es estacionario (p-ADF 0.87 en el fold 1) y en que esa MI identifica la época, no una regla. Demostrarlo con MI contra un nulo de **bloques** barajados, o mostrando que no se sostiene en el fold siguiente.
6. **Regímenes por fold.**
   - **Hallazgo:** las fronteras salen del train del fold 4, que incluye la validación y el test del fold 1.
   - **Cambio:** declarar que solo se usan para describir y mostrar que el train del fold 1 ya contiene al menos 4 de ellos (R1 a R5 terminan antes del 2021-11-08). Justificar el umbral de 60%, por ejemplo con cuántos regímenes salen con 40%.
7. **Diferencia de filas.**
   - **Hallazgo:** el CSV trae 75,588 filas, el PDF dice 75,558 y quedan 75,082 barras únicas tras quitar 506 duplicados.
   - **Cambio:** confirmar con el profe que el archivo de Canvas es el correcto. Ya está documentado en el docstring de `data.py`; falta la confirmación.
8. **Justificar q y R del filtro de Kalman.**
   - **Cambio:** un párrafo en el reporte con la cuenta de la vida media: k = 1 − 0.5^(1/24) y q = k²/(1 − k), con R_t = varianza móvil de 24 barras.
   - **Razón:** en el reporte es la componente de creencia B_t de la formulación de Powell.

## OPCIONAL

9. **Ventanas por tiempo.**
   - **Hallazgo:** las ventanas cuentan barras, y desde abril de 2025 falta casi a diario la barra de 00:00 (230 huecos de 1 h en 2025). `vol_24` cubre unas 25 h en valid y test del fold 4 contra 24 h en train.
   - **Cambio:** declararlo como limitación, o usar `rolling("24h")`. Es un efecto pequeño, pero el profe puede preguntarlo.
10. **Creencia de régimen causal.** Por ejemplo, un HMM de 2–3 estados ajustado solo con el train de cada fold, como variable de B_t.
