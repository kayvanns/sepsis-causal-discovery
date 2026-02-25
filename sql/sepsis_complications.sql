WITH sepsis AS (
  SELECT
    s.subject_id,
    i.hadm_id,
    s.stay_id,
    i.intime  AS icu_intime,
    i.outtime AS icu_outtime,
    s.suspected_infection_time AS sepsis_onset_time,
    s.sofa_score
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3` s
  JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
    ON s.stay_id = i.stay_id
  WHERE s.sepsis3 = TRUE
),
/* ---------- AKI within 0-24h ---------- */
aki_24h AS (
  SELECT
    k.stay_id,
    MAX(k.aki_stage_smoothed) AS aki_24h_max_stage,
    MAX(CASE WHEN k.aki_stage_smoothed >= 1 THEN 1 ELSE 0 END) AS aki_24h_any
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
  JOIN sepsis s ON k.stay_id = s.stay_id
  WHERE k.charttime >= s.sepsis_onset_time
    AND k.charttime < TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  GROUP BY k.stay_id
),

/* ---------- AKI after 24h (24h -> ICU outtime) ---------- */
aki_post24h AS (
  SELECT
    k.stay_id,
    MAX(k.aki_stage_smoothed) AS aki_post24h_max_stage,
    MAX(CASE WHEN k.aki_stage_smoothed >= 1 THEN 1 ELSE 0 END) AS aki_post24h_any
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
  JOIN sepsis s ON k.stay_id = s.stay_id
  WHERE k.charttime >= TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
    AND k.charttime <= s.icu_outtime
  GROUP BY k.stay_id
),

/* ---------- Mechanical ventilation within 0-24h ----------
   We treat "overlap with window" as true if the vent interval overlaps [onset, onset+24h).
*/
mechvent_24h AS (
  SELECT
    v.stay_id,
    1 AS mechvent_24h_any
  FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
  JOIN sepsis s ON v.stay_id = s.stay_id
  WHERE ventilation_status IN ('InvasiveVent', 'Tracheostomy')
    AND v.starttime < TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
    AND v.endtime   > s.sepsis_onset_time
  GROUP BY v.stay_id
),

/* ---------- Mechanical ventilation after 24h ----------
   Overlap with [onset+24h, ICU outtime]
*/
mechvent_post24h AS (
  SELECT
    v.stay_id,
    1 AS mechvent_post24h_any
  FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
  JOIN sepsis s ON v.stay_id = s.stay_id
  WHERE ventilation_status IN ('InvasiveVent', 'Tracheostomy')
    AND v.starttime < s.icu_outtime
    AND v.endtime   > TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR)
  GROUP BY v.stay_id
)

SELECT
  s.*,
  /* AKI flags */
  COALESCE(a24.aki_24h_any, 0)        AS aki_24h_onset,
  COALESCE(a24.aki_24h_max_stage, 0)  AS aki_24h_onset_stage,
  COALESCE(ap.aki_post24h_any, 0)     AS aki_post24h,
  COALESCE(ap.aki_post24h_max_stage, 0) AS aki_post24h_stage,

  /* Mech vent flags */
  COALESCE(m24.mechvent_24h_any, 0)      AS mechvent_24h_onset,
  COALESCE(mp.mechvent_post24h_any, 0)   AS mechvent_post24h

FROM sepsis s
LEFT JOIN aki_24h a24        ON a24.stay_id = s.stay_id
LEFT JOIN aki_post24h ap     ON ap.stay_id  = s.stay_id
LEFT JOIN mechvent_24h m24   ON m24.stay_id = s.stay_id
LEFT JOIN mechvent_post24h mp ON mp.stay_id = s.stay_id;