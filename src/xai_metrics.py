import numpy as np

def localization_metrics(attribution,bbox_mask):
    a=np.maximum(np.asarray(attribution,dtype=float),0); total=a.sum()
    if total<=0: raise ValueError("Attribution map has zero positive energy.")
    a=a/total; bbox=np.asarray(bbox_mask)>0; area=float(bbox.mean()); energy=float(a[bbox].sum()); enrichment=energy/area if area>0 else np.nan; peak=np.unravel_index(np.argmax(a),a.shape); pointing=int(bbox[peak]); p=a[a>0]; entropy=float(-np.sum(p*np.log(p))/np.log(a.size))
    return {"bbox_area_fraction":area,"bbox_energy_fraction":energy,"energy_enrichment":enrichment,"pointing_game":pointing,"normalized_entropy":entropy}
