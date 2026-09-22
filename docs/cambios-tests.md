# Registro completo de cambios en los tests

Comparación con la suite original de project. No se modificaron ecuaciones, solvers ni tolerancias científicas. Los nombres siguientes identifican funciones; se indican los casos parametrizados cuando corresponden.

## Archivo excluido: tests/test_delivery.py

Se excluye entero por depender de entregas y operaciones de Git. Sus 24 funciones eran:

| Test eliminado | Motivo y cobertura anterior o sustituta |
|---|---|
| `test_the_audit_passes_on_the_repository_as_it_stands` | Auditaba el inventario histórico del curso; sustituido por la auditoría autónoma de distribución y procedencia. |
| `test_the_recorded_audit_matches_a_fresh_run` | Comparaba el informe histórico de auditoría con una corrida nueva; ese informe no se publica. |
| `test_the_four_preserved_failures_are_still_failing_and_still_declared` | Vigilaba los cuatro rechazos preservados; ahora se comparan con artifacts/limitations.json en la auditoría autónoma. |
| `test_an_artifact_without_provenance_fails_the_audit` | Detectaba artefactos sin procedencia; cubierto por missing_record del nuevo control científico. |
| `test_a_record_without_its_artifact_fails_the_audit` | Detectaba registros sin archivo; lo verifica el nuevo inventario y su control missing. |
| `test_a_missing_declared_input_fails_the_audit` | Detectaba entradas ausentes; cubierto por missing_input del nuevo control científico. |
| `test_a_changed_artifact_fails_the_digest_gate` | Detectaba cambios de bytes; cubierto por changed del nuevo control de distribución y los hashes históricos. |
| `test_an_undeclared_failing_check_fails_the_audit` | Detectaba rechazos no declarados; cubierto por undeclared_failure. |
| `test_a_declared_failure_that_starts_passing_fails_the_audit` | Detectaba declaraciones de rechazo obsoletas; cubierto por stale_failure. |
| `test_a_milestone_without_an_acceptance_script_fails_the_audit` | Exigía scripts de aceptación por hito del curso; se excluyen hitos y scripts. |
| `test_an_experiment_that_produces_nothing_fails_the_audit` | Exigía correspondencia de experimentos con el inventario histórico de entregas; no se traslada ese catálogo. Se conserva procedencia de los resultados publicados. |
| `test_the_page_and_the_pdf_are_declared_and_match_their_digests` | Auditaba página y PDF de la entrega del curso, excluidos; el nuevo inventario cubre el PDF vigente. |
| `test_the_page_rebuilds_byte_for_byte_over_unchanged_evidence` | Reproducía la página del curso byte por byte; esa página y su generador se excluyen. |
| `test_no_number_in_the_page_is_typed_rather_than_read` | Comparaba números de la página del curso con datos; reemplazado para el paper por regeneración de numbers.tex y CSV. |
| `test_the_page_shows_every_failing_check_and_the_negative_results` | Exigía rechazos visibles en la página del curso; la distribución los declara en limitations.json y docs/resultados.md. |
| `test_the_page_makes_no_network_request` | Prohibía solicitudes de red en la página del curso; la página no se publica. |
| `test_a_preserved_failure_without_spanish_wording_stops_the_build` | Exigía traducciones de rechazos en el generador de la entrega; el generador se excluye. |
| `test_the_documents_carry_provenance_naming_their_sources` | Exigía procedencia de HTML/PDF del curso; sustituidos por el inventario final y los controles del suplemento vigente. |
| `test_the_reproduction_guide_exists_and_names_the_commands_it_claims` | Comprobaba comandos propios del curso; la guía se reescribe y sus comandos de verificación se ejecutan. |
| `test_the_staged_bytes_gate_catches_what_git_status_hides` | Creaba un repositorio y manipulaba el índice para detectar conversiones de líneas; excluido para no ejecutar Git. Se conserva el control negativo de bytes sin Git. |
| `test_the_staged_bytes_gate_says_so_when_it_cannot_run` | Probaba la puerta del índice de Git fuera de un repositorio; esa puerta se excluye. |
| `test_nested_artifacts_require_provenance` | Auditaba HTML/PDF anidados de entregas; no se publican. El inventario final cubre recursivamente los archivos distribuidos. |
| `test_staged_bytes_detects_a_missing_tracked_file` | Inicializaba Git para detectar archivos rastreados ausentes; sustituido por el control missing sin Git. |
| `test_invalid_manifest_reports_a_failed_schema_gate` | Probaba el esquema del inventario del curso; cubierto por invalid_schema y los tests originales de provenance. |

## Tests reemplazados de tests/test_manuscript.py

Se retira el control del borrador reports/manuscrito-hito-8.md y su PDF.

| Test anterior eliminado | Motivo y cobertura sustituta |
|---|---|
| `test_the_published_anchors_appear_with_the_measured_deviations` | Buscaba anclas y desviaciones en el borrador Markdown. Ahora se regeneran todas las macros y se comprueban las referencias del LaTeX vigente. |
| `test_the_error_budget_numbers_are_the_recorded_ones` | Buscaba el antiguo presupuesto de errores en Markdown. Lo sustituyen la regeneración de cifras y la auditoría de sensibilidad actual. |
| `test_the_external_comparisons_are_the_recorded_ones` | Buscaba cifras de contrastes en el borrador. Ahora se comparan todas las macros generadas; los tests científicos de contrastes permanecen. |
| `test_the_extremal_shift_is_quoted_from_the_measurement` | Buscaba los valores de extremo en Markdown. Quedan cubiertos por numbers.tex regenerado y el CSV. |
| `test_the_sign_locus_table_is_the_recorded_one` | Buscaba la tabla de cambio de signo en el borrador. Queda cubierta por la regeneración de numbers.tex. |
| `test_the_manuscript_declares_every_failing_gate` | Exigía nombres de gates en el borrador. Se sustituye por declaraciones estructuradas en limitations.json, auditadas contra el manifiesto. |
| `test_the_manuscript_does_not_claim_what_is_already_published` | Buscaba citas y frases de prioridad del borrador. El control vigente conserva las tres anclas; no pretende probar prioridad científica por búsqueda textual. |
| `test_the_ergosurface_limitation_is_stated` | Exigía los dos radios discrepantes en el borrador. El nuevo test los exige en main.tex. |
| `test_the_pdf_was_built_from_the_same_run` | Sólo exigía existencia y tamaño del PDF antiguo; no probaba igualdad de corrida. El nuevo control comprueba PDF vigente/figuras y el inventario fija sus bytes. |
| `test_no_placeholder_survived` | Buscaba placeholders en Markdown. Ahora se comprueba main.tex y que el correo esté vacío. |

## Tests modificados

| Test | Motivo y cobertura |
|---|---|
| `test_egb_rotating_paper_profiles.py::test_separable_curves_and_printed_tick_calibration` | Misma extracción de diez lecturas y misma calibración; se salta únicamente si falta la imagen externa. |
| `test_egb_rotating_paper_profiles.py::test_red_finite_coupling_curve_rejects_closed_vacuum_profile` | Conserva el control negativo contra Myers–Perry y la tolerancia de tres píxeles; se salta únicamente si falta la imagen externa. |
| `test_recorded_digests.py::test_every_recorded_digest_matches_the_bytes_on_disk` | Conserva hashes y excepción documental original; permite sólo la ausencia explícita de la imagen externa de hash conocido. Si está presente, la verifica. |
| `test_myers_perry_benchmark.py::test_generation_repeats_data_and_hashes_in_local_copy` | Deja de copiar prompts, que no son entradas del cálculo. Conserva ambas ejecuciones, igualdad byte por byte del resultado, dos registros de procedencia, checks y todos los hashes. |

## Tests nuevos

| Archivo y test | Motivo y comprobación |
|---|---|
| `test_manuscript.py::test_current_manuscript_macros_and_citations_resolve` | Ejecuta el checker del LaTeX vigente: macros definidas/usadas, citas resolubles y política de decimales. |
| `test_manuscript.py::test_saved_numbers_equal_regenerated_numbers` | Regenera todas las macros y tablas desde los resultados guardados en un directorio temporal y compara numbers.tex. |
| `test_manuscript.py::test_solution_table_contains_the_actual_observables` | Compara CSV regenerado y publicado; verifica cada observable contra las soluciones originales del JSON. |
| `test_manuscript.py::test_sensitivity_audit_regenerates_and_encloses_alternatives` | Regenera la auditoría, verifica sus hashes y que la envolvente contenga cada ajuste alternativo. |
| `test_manuscript.py::test_current_paper_preserves_external_anchors_and_limitation` | Comprueba las tres referencias externas, ambos radios discrepantes, ausencia de placeholders y correo vacío. |
| `test_manuscript.py::test_current_pdf_and_generated_figures_are_present` | Verifica cabecera y tamaño del PDF vigente y figuras; exige que las figuras se regeneren como PDF. No prueba por sí solo correspondencia semántica del PDF con LaTeX. |
| `test_distribution.py::test_published_distribution_is_intact` | Ejecuta la auditoría de todos los bytes publicados, esquema, entradas, cobertura científica y rechazos declarados. |
| `test_distribution.py::test_distribution_audit_rejects_corruption` | Cuatro casos: archivo alterado, ausente, agregado sin inventariar y ruta fuera de la raíz. Cada uno debe fallar. |
| `test_distribution.py::test_scientific_inventory_rejects_broken_provenance` | Cinco casos: entrada ausente, rechazo no declarado, declaración obsoleta, esquema inválido y resultado sin procedencia. |
| `test_recorded_digests.py::test_only_the_documented_external_image_may_be_absent` | Sólo acepta la ausencia de la imagen con el hash histórico exacto; rechaza una imagen presente corrupta y otro hash declarado. |
| `test_recorded_digests.py::test_missing_scientific_source_is_still_an_error` | Una fuente científica ausente sigue haciendo fallar el verificador; la excepción no se generaliza. |

Los demás tests se conservan byte por byte. `test_git_is_configured_never_to_rewrite_bytes` sólo lee `.gitattributes`: no ejecuta Git. Las referencias ficticias a prompts en fixtures de esquema no requieren esos archivos.
