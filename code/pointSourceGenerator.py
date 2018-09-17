"""
Script to generate a simulation of the full microwave sky adding point
sources in the positions and with the temperature flux, polarization 
angle and polarization degree desired. Temperature, Q and U and E and B mode
maps are produced.

The microwave sky simulation is done using the PySM code developed by
B. Thorne, J. Dunkley, D. Alonso and S. Naess that can be found at
https://github.com/bthorne93/PySM_public.

@author: Patricia Diego Palazuelos
"""
import argparse
import textwrap
import numpy as np
import healpy as hp
from pysm.common import convert_units
import os
import pysm
import sys
import warnings
from healpy.projector import CartesianProj
from pysm.nominal import models
import numpy.random as random

"""
Ignore warnings to avoid printing the 'divide by zero' Runtime Warning when
polar coordinates are calculated for the point-like sources
"""
if not sys.warnoptions:
   warnings.simplefilter("ignore")
    

def initial():
    """
    Function to read input arguments from command line and to offer help to user.
    """
    parser = argparse.ArgumentParser(
    prog='pointSourceGenerator.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    pointSourceGenerator
    -------------
    Returns a flat patch of the E- and B-mode polarization maps of the microwave
    sky, with a point-like source of the desired properties at its center. Only 
    the 30 GHz, 100 GHz and 353 GHz channels of the Planck satellite, and the 
    three noise levels defined in the work are simulated.
    '''))
    parser.add_argument('-p', metavar='coordinates',action='store', default="thetaphi", type=str,
                        help='Format for the input source coordinates: longitude and latitude (\'lonlat\') or spherical theta and phi (\'thetaphi\'). \nDefault=thetaphi')
    parser.add_argument('-v',action='store_true',
                        help='Verbose mode. Default = False')
    parser.add_argument('nu', metavar='frequency', type=int, nargs=1,
                        help='Frequency (GHz) of the Planck channel to simulate.')
    parser.add_argument('patchName', metavar='patch_name', type=str, nargs=1,
                        help='Name for the output patches.')
    parser.add_argument('source_position', metavar='position', type=str, nargs=1,
                        help='Introduce source coordinates as: \'theta1,phi1 theta1,phi1...\'.')
    parser.add_argument('flux', metavar='flux', type=str, nargs=1,
                        help='Flux (Jy) for the point-like source. Introduce as:\'flux1 flux2 ...\'.')
    parser.add_argument('polarization_degree', metavar='polarization_degree', type=str, nargs=1,
                        help='Polarization degree (Q²+U²)^(1/2)/I of each source.')
    parser.add_argument('polarization_angle', metavar='polarization_angle', type=str, nargs=1,
                        help='Polarization angle of each source in degrees.')  
    parser.add_argument('noise_level', metavar='noise_level', type=int, nargs=1,
                        help='Index indentifying the three noise levels used in the work. 1 is the highest and 3 is the lowest.')
   
    args      = parser.parse_args()
    return args
  
def gaussianNoiseFlat(s,l):
    noise=np.zeros((int(l),int(l)),dtype=float)
    for i in np.arange(0,l,1,dtype=int):
        for j in np.arange(0,l,1,dtype=int):
            #gaussian distribution mean=0 sigma=s
            noise[i,j]=random.normal(0.,s)
    
    return noise

noiseLevels=[[210,50,5],[120,50,5],[440,50,5]]
pixelization=[[512,5.5],[1024,10/3],[2048,10/3]]
#phenomenological factor to match the definitions in the plane and in the sphere
facEB=0.2850964412168842
#read arguments from terminal call 
args=initial()
coordinates=args.p
k=args.noise_level[0]
verbose=args.v
nu=args.nu[0]
mapName=args.patchName[0]
position=args.source_position[0]
flux=args.flux[0]
polDegree=args.polarization_degree[0]
polAngle=args.polarization_angle[0]

if nu not in [30,100,353]:
    print('Frequency not supported.')
    exit()
    
if nu==30:
    n=0
elif nu==100:
    n=1
else:
    n=2
    
if k not in [1,2,3]:
    print('Index for noise level not supported.')
    exit()    

words1=flux.split()
words2=position.split()
words3=polDegree.split()
words4=polAngle.split()
#check equal number of inputs for each property
if not (len(words1)==len(words2)==len(words3)==len(words4)):
    print('Missing properties for source.')
    exit()
    
#split properties of each source
f=np.zeros_like(words1,dtype=float)
p=np.zeros_like(words3,dtype=float)
phi=np.zeros_like(words4,dtype=float)
pos=[]
for i in np.arange(0,len(words1),1):
    f[i]=float(words1[i])
    #check polarization degree is lower or equal to one
    a=float(words3[i])
    if a>1:
        print('Polarization degree must be lower or equal to one.')
        exit()
    else:
        p[i]=a
    #check polarization angle is between [0,180] degrees
    b=float(words4[i])
    if b>=0 and b<180:
        phi[i]=b
    else:
        print('Polarization angle must be between [0,180) degress.')
        exit()
        
    #split position coordinates
    c=words2[i].split(',')
    pos.append(np.array([float(c[0]),float(c[1])]))
     
#produce maps
nside=pixelization[n][0]
lsize=int(nside/4)
centralpix=int(lsize/2)
fwhmpix=pixelization[n][1]
Apix=hp.nside2pixarea(nside)#square radians
fwhmrad=np.sqrt(Apix)*fwhmpix
sigma=fwhmpix/2.355482
#sky config with the principal models for each component
sky_config={
        'synchrotron':models("s1",nside),
        'dust':models("d1",nside),
        'freefree':models("f1",nside),
        'ame':models("a1",nside),
        'cmb':models("c1",nside)
        }

sky=pysm.Sky(sky_config) 
#unit conversion factor
mapFactor=convert_units("uK_RJ","uK_CMB",nu)
sourceFactor=convert_units("Jysr","uK_CMB",nu)
allmaps=sky.signal()(nu)

if verbose:
    print('Microwave sky simulated')

T=np.array(allmaps[0], dtype=float)*mapFactor
Q=np.array(allmaps[1], dtype=float)*mapFactor
U=np.array(allmaps[2], dtype=float)*mapFactor    
#smoothing with gaussian psf
t=hp.smoothing(map_in=T,fwhm=fwhmrad, verbose=False)
q=hp.smoothing(map_in=Q,fwhm=fwhmrad, verbose=False)
u=hp.smoothing(map_in=U,fwhm=fwhmrad, verbose=False)
if verbose:
    print('Maps smoothed')

alms=hp.map2alm([t,q,u])
emap=hp.alm2map(alms[1],nside=nside,verbose=False)
bmap=hp.alm2map(alms[2],nside=nside,verbose=False)

def vec2pix(x,y,z):
    """
    Auxiliar function for projection.
    """
    return hp.vec2pix(nside,x,y,z)

epatches=[]
bpatches=[]
for h in np.arange(0,len(pos),1,dtype=int):
    #coordinate limits that define each patch
    #theta->pos[i][0], phi->pos[i][1], degrees
    #lon->pos[i][0], lat->pos[i][1], degrees
    if coordinates=='thetaphi':
        lat=90-pos[h][0]
    else:
        lat=pos[h][0]
 
    if pos[h][1]>180:
        lon=pos[h][1]-360
    else:
        lon=pos[h][1]
    
    #patches of 12.8ªx12.8ª    
    if lat+6.4>90.:
        a=(lat+6.4)-90.
        latra=[a-90, lat-6.4]
    elif lat-6.4<-90:
        b=(lat-6.4)+90
        latra=[lat+6.4,90+b]
    else:
        latra=[lat-6.4, lat+6.4]
        
    if lon+6.4>180.:
        a=(lon+6.4)-180
        lonra=[a-180.,lon-6.4]
    elif lon-6.4<-180:
        b=(lon-6.4)+180.
        lonra=[lon+6.4, 180+b]
    else:
        lonra=[lon-6.4, lon+6.4]
    
    proj=CartesianProj(xsize=lsize,ysize=lsize,lonra=lonra, latra=latra)
    eflat=proj.projmap(emap,vec2pix)
    bflat=proj.projmap(bmap,vec2pix)
    #generate point-like source
    P=(p[h]*facEB*f[h]*sourceFactor)/(Apix)
    psE=np.zeros((lsize,lsize),dtype=float)
    psB=np.zeros((lsize,lsize),dtype=float)
    for i in np.arange(0, lsize,1, dtype=int):
        for j in np.arange(0,lsize,1, dtype=int):
            #x,y con signo correspondiente a cada cuadrante 
            y=i-centralpix
            x=j-centralpix
            r2=x**2+y**2
            if x==0. and y==0.:
                theta=0.
                c=0.
                exp=0.
            else:
                theta=np.arctan(y/x)
                if (x<0 and y>0) or (x<0 and y<0):
                    theta=np.pi+theta
                elif x>0 and y<0:
                    theta=2*np.pi+theta
                c=np.cos(2*theta)   
                z=r2/(2*sigma**2)
                exp=(sigma**2*np.exp(-z)*(1+z)-1)/(4*np.pi**2*r2)
                
            
            s=np.sin(2*theta)
            psE[i,j]=(np.cos(2*np.deg2rad(phi[h]))*c+np.sin(2*np.deg2rad(phi[h]))*s)*exp*P
            psB[i,j]=(np.cos(2*np.deg2rad(phi[h]))*s-np.sin(2*np.deg2rad(phi[h]))*c)*exp*P
                
    #add point-like source 
    em=np.add(eflat,psE)
    bm=np.add(bflat,psB)
    #add noise
    en=np.add(em,gaussianNoiseFlat(noiseLevels[n][k-1],lsize))
    bn=np.add(bm,gaussianNoiseFlat(noiseLevels[n][k-1],lsize))
    epatches.append(en)
    bpatches.append(bn)
     
if verbose:
    print('Patches projected')

newpath = 'patches/'+mapName 
if not os.path.exists(newpath):
    os.makedirs(newpath)    
    
np.save(newpath+'/'+mapName+'_E',epatches)
np.save(newpath+'/'+mapName+'_B',bpatches)

if verbose:
    print('Patches saved')
