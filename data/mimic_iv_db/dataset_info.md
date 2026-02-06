# Dataset Link

Wget from [here](https://physionet.org/content/mimic-iv-ed-demo/2.2/) with

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-ed-demo/2.2/
mv physionet.org/files/mimic-iv-ed-demo/2.2/ed ./
rm -r physionet.org/
cd ed
gunzip *.gz
```

then, manually modified some key values to create nesting, namely:

> - *intime/outtime* -> *in_time/out_time* in edstays.csv
> - *etccode/etcdescription/charttime/name* -> *etc_code/etc_description/chart_time_medrecon/name_medrecon* in medrecon.csv
> - *charttime/name* -> *chart_time_pyxis/name_pyxis* in pyxis.csv
> - *heartrate,resprate,o2sat,chiefcomplaint* -> *heart_rate,resp_rate,o2_sat,chief_complaint* in triage.csv
> - *heartrate,resprate,o2sat,charttime* -> *heart_rate,resp_rate,o2_sat,chart_time_vitalsign* in vitalsign.csv

lastly, generate the merged csv with the script *mimic_iv.py*