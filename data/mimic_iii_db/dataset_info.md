# Dataset Link

Wget from [here](https://physionet.org/content/mimiciii-demo/1.4/) with

```bash
wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/
mv physionet.org/files/mimiciii-demo/1.4/ ./ed
rm -r physionet.org/
cd ed
gunzip *.gz
```

then, manually modified some key values to create nesting, namely:

> - *admittime/dischtime/deathtime/edregtime/edouttime/has_chartevents_data* -> *admit_time/disch_time/death_time/ed_reg_time/ed_out_time/has_chart_events_data* in ADMISSIONS.csv
> - *icustay_id/dbsource/eventtype/intime/outtime* -> *icu_stay_id/db_source/event_type/in_time/out_time* in TRANSFERS.csv
> - *transfer_time* -> *transfer_time* in SERVICES.csv
> *icustay_id/starttime/endtime/itemid/valueuom/locationcategory/storetime/cgid/orderid/linkorderid/ordercategoryname/secondaryordercategoryname/ordercategorydescription/isopenbag/continueinnextdept/cancelreason/statusdescription/comments_editedby/comments_canceledby/comments_date* -> *icu_stay_id/start_time/end_time/item_id/value_uom/location_category/store_time/cg_id/order_id/link_order_id/order_category_name/secondary_order_category_name/order_category_description/is_open_bag/continue_in_next_dept/cancel_reason/status_description/comments_edited_by/comments_canceled_by/comments_date* in PROCEDUREEVENTS_MV.csv
> - *icustay_id/startdate/enddate* -> *icu_stay_id/start_date/end_date* in PRESCRIPTIONS.csv
> - *icustay_id/charttime/itemid/valueuom/storetime/cgid/stopped/newbottle/iserror* -> *icu_stay_id/chart_time/item_id/value_uom/store_time/cg_id/new_bottle/is_error* in OUTPUTEVENTS.csv
> - NOTEEVENTS.csv was empty so we deleted it
> - *chartdate,charttime,spec_itemid,org_itemid,ab_itemid* -> *chart_date/chart_time/spec_item_id/org_item_id/ab_item_id* in MICROBIOLOGYEVENTS.csv
> - *itemid/charttime/valuenum/valueuom* -> *item_id/chart_time/value_num/value_uom* in LABEVENTS.csv
> - *icustay_id/starttime/endtime/itemid/amountuom/rateuom/storetime/cgid/orderid/linkorderid/ordercategoryname/secondaryordercategoryname/ordercomponenttypedescription/ordercategorydescription/patientweight/totalamount/totalamountuom/isopenbag/continueinnextdept/cancelreason/statusdescription/comments_editedby/comments_canceledby/comments_date/originalamount/originalrate* -> *icu_stay_id/start_time/end_time/item_id/amount_uom/rate_uom/store_time/cg_id/order_id/link_order_id/order_category_name/secondary_order_category_name/order_component_type_description/order_category_description/patient_weight/total_amount/total_amount_uom/is_open_bag/continue_in_next_dept/cancel_reason/status_description/comments_edited_by/comments_canceled_by/original_amount/original_rate* in INPUTEVENTS_MV.csv
> - *icustay_id/charttime/itemid/amountuom/rateuom/storetime/cgid/orderid/linkorderid/newbottle/originalamount/originalamountuom/originalroute/originalrate/originalrateuom/originalsite* -> *icu_stay_id/chart_time/item_id/amount_uom/rate_uom/store_time/cg_id/order_id/link_order_id/new_bottle/original_amount/original_amount_uom/original_route/original_rate/original_rate_uom/original_site* in INPUTEVENTS_CV.csv
> - *icustay_id/dbsource/first_careunit/last_careunit/first_wardid/last_wardid/intime/outtime* -> 
*icu_stay_id/db_source/first_care_unit/last_care_unit/first_ward_id/last_ward_id/in_time/out_time* in ICUSTAYS.csv 
> - *icustay_id/itemid/charttime/storetime/cgid/valueuom/resultstatus* -> *icu_stay_id/item_id/chart_time/store_time/cg_id/value_uom/result_status* in DATETIMEECVENTS.csv
> - *itemid/dbsource/linksto/unitname/conceptid* -> *item_id/db_source/links_to/unit_name/concept_id* in D_ITEMS.csv
> - *sectionrange/sectionheader/subsectionrange/subsectionheader/codesuffix/mincodeinsubsection/maxcodeinsubsection* -> *section_range/section_header/subsection_range/subsection_header/code_suffix/min_code_in_subsection/max_code_in_subsection* in D_CPT.csv
> - *costcenter/chartdate/sectionheader/subsectionheader* -> *cost_center/chart_date/section_header/subsection_header/description* in CPTEVENTS.csv
> - *icustay_id/itemid/charttime/storetime/cgid/valuenum/valueuom/resultstatus* -> *icu_stay_id/item_id/chart_time/store_time/cg_id/value_num/value_uom/result_status* in CHARTEVENTS.csv
> - *submit_wardid/submit_careunit/curr_wardid/curr_careunit/callout_wardid/discharge_wardid/createtime/updatetime/acknowledgetime/outcometime/firstreservationtime/currentreservationtime* -> *submit_ward_id/submit_care_unit/curr_ward_id/curr_care_unit/callout_ward_id/discharge_ward_id/create_time/update_time/acknowledge_time/outcome_time/first_reservation_time/current_reservation_time* in CALLOUT.csv

lastly, generate the merged csv with the script *mimic_iii.py*
