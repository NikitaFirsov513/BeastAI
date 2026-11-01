"""Find geometrically related images without claiming recovered flight/source IDs."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,time
import cv2,numpy as np
import waid
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/roe/audit';OUT=ROOT/'data/roe/grouping'


def features(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    scale=min(1.,640/max(gray.shape));gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else gray
    detector=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02)
    points,desc=detector.detectAndCompute(gray,None)
    if desc is None or len(desc)<2:return {'points':np.empty((0,2),np.float32),'desc':None,'shape':gray.shape}
    return {'points':np.float32([p.pt for p in points]),'desc':np.float32(desc),'shape':gray.shape}


def relation(a,b,matcher=None):
    if a['desc'] is None or b['desc'] is None:return None
    if matcher is None:
        matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
        matches=matcher.knnMatch(a['desc'],b['desc'],k=2)
    else:
        matches=matcher.knnMatch(a['desc'],k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance]
    if len(good)<24:return None
    source=np.float32([a['points'][m.queryIdx] for m in good]);target=np.float32([b['points'][m.trainIdx] for m in good])
    transform,mask=cv2.findHomography(source,target,cv2.RANSAC,3.,maxIters=4000,confidence=.995)
    if transform is None or mask is None:return None
    mask=mask.ravel().astype(bool);inliers=int(mask.sum());ratio=inliers/len(good)
    if inliers<20 or ratio<.5:return None
    coverage_a=cv2.contourArea(cv2.convexHull(source[mask]))/(a['shape'][0]*a['shape'][1]);coverage_b=cv2.contourArea(cv2.convexHull(target[mask]))/(b['shape'][0]*b['shape'][1])
    if min(coverage_a,coverage_b)<.04:return None
    error=np.linalg.norm(cv2.perspectiveTransform(source[mask,None,:],transform)[:,0,:]-target[mask],axis=1)
    return {'good_matches':len(good),'inliers':inliers,'inlier_fraction':ratio,'source_feature_coverage':coverage_a,'target_feature_coverage':coverage_b,'median_reprojection_error_pixels':float(np.median(error))}


def main():
    cv2.setNumThreads(4);cv2.setRNGSeed(42)
    audit=json.loads((DATA/'summary.json').read_text(encoding='utf-8'));OUT.mkdir(parents=True,exist_ok=True)
    images={item['name']:cv2.imdecode(np.fromfile(DATA/'images'/item['name'],dtype=np.uint8),cv2.IMREAD_COLOR) for item in audit['sample']}
    assert all(image is not None for image in images.values())
    start=time.perf_counter();normal={name:features(image) for name,image in images.items()};flipped={name:features(cv2.flip(image,1)) for name,image in images.items()}
    # Positive numerical control: a known flip, rotation and brightness change of one image.
    first=next(iter(images.values()));flipped_image=cv2.flip(first,1);h,w=flipped_image.shape[:2]
    transformed=cv2.warpAffine(flipped_image,cv2.getRotationMatrix2D((w/2,h/2),7.,.92),(w,h));transformed=cv2.convertScaleAbs(transformed,alpha=1.08,beta=4)
    selfcheck=relation(features(cv2.flip(first,1)),features(transformed))
    assert selfcheck is not None,'Known related-image positive control failed'
    assert relation(features(np.zeros_like(first)),features(first)) is None
    names=list(images);parent={name:name for name in names};pairs=[]
    def find(name):
        while parent[name]!=name:parent[name]=parent[parent[name]];name=parent[name]
        return name
    for i,a in enumerate(names):
        for b in names[i+1:]:
            candidates=[]
            for orientation,feature in [('normal',normal[b]),('horizontal_flip',flipped[b])]:
                match=relation(normal[a],feature)
                if match:candidates.append({'a':a,'b':b,'target_orientation':orientation,**match})
            if candidates:
                best=max(candidates,key=lambda m:m['inliers']);pairs.append(best);parent[find(b)]=find(a)
        if i%8==0:print(f'[{i+1}/32] pair comparisons processed',flush=True)
    groups=defaultdict(list)
    for name in names:groups[find(name)].append(name)
    result={'input_image_sha256':{item['name']:item['sha256'] for item in audit['sample']},'images':len(names),'groups':list(groups.values()),'nontrivial_groups':[g for g in groups.values() if len(g)>1],'related_pairs':pairs,'selfcheck':selfcheck,'elapsed_seconds':time.perf_counter()-start,'method':'SIFT, normal/mirrored targets, ratio 0.72, RANSAC homography 3px, >=20 inliers, >=0.5 inlier fraction, >=4% feature hull coverage in each image','opencv':cv2.__version__,'limitations':['Visual relations are candidate same-source/same-scene groups, not recovered flight IDs.','An absent match does not prove images come from different flights or originals.','This scan covers only the 32 frozen diagnostic images. No valid train/test split is established by this scan alone.']}
    waid.save_json(OUT/'sample_groups.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('related_pairs','groups')}),flush=True)

if __name__=='__main__':main()
