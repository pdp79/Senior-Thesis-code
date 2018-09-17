"""
Applies the filter to the point-like source located at the given coordinates 
and returns its polarization angle and magnitude.

@author: Patricia Diego Palazuelos
"""

"""
se la frecuencia y el ruido y me dan ya el parche
tengo que filtrar
determinar que zona es
y asignarle los errores
"""

import argparse
import textwrap
import numpy as np
import healpy as hp
import scipy.fftpack as ft
from scipy.interpolate import CubicSpline
from pysm.common import convert_units
import sys
import warnings

"""
Ignore warnings to avoid printing the 'divide by zero' Runtime Warning when
polar coordinates are calculated for the filter images
"""
if not sys.warnoptions:
   warnings.simplefilter("ignore")
    
def initial():
    """
    Function to read input arguments from command line and to offer help to user.
    """
    parser = argparse.ArgumentParser(
    prog='applyFilter.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    applyFilter
    -------------
    Applies the filter to the given patches and returns the polarization angle
    (degrees) and intensity (Jy), with their associated errors.
    '''))
    parser.add_argument('patchName', metavar='patch_name', type=str, nargs=1,
                        help='Name for the input patches.')
    parser.add_argument('nu', metavar='frequency', type=int, nargs=1,
                        help='Frequency (GHz) channel the patches come from.')
    parser.add_argument('calibf_name', metavar='calibration_function_name', type=str, nargs=1,
                        help='Name for the calibration function in .npy format. For example:"nside 512 fw 5.5 dphi60 dR0.05"')
    parser.add_argument('noise_level', metavar='noise_level', type=int, nargs=1,
                        help='Index indentifying the three noise levels used in the work. 1 is the highest and 3 is the lowest.')
    parser.add_argument('-v', action='store_true',
                        help='Verbose mode. Default = False')
    args      = parser.parse_args()
    return args


def getPointsForInterpolation(i,j,phiout,G,PHI):
    """
    Auxiliar function for the interpolation function.
    """
    low=7
    up=8
    lowerL=i-low
    upperL=i+up
    if lowerL<0:
        lowerL=lowerL+len(PHI[:,0])
    if upperL>(len(PHI[:,0])-1):
        upperL=upperL-len(PHI[:,0])
    #avoid returning an empty array when upperL=0
    if upperL==0:
        upperL=upperL+1
    if lowerL>upperL: 
        if i>upperL:
            phiout=phiout-np.pi
            
        aux=np.array(np.ones_like(PHI[lowerL:,j]))*-np.pi            
        pointsPhi=np.concatenate((np.add(PHI[lowerL:,j],aux),PHI[:upperL,j]))
        pointsG=np.concatenate((G[lowerL:,j],G[:upperL,j]))
    else:
        pointsPhi=PHI[lowerL:upperL,j]
        pointsG=G[lowerL:upperL,j]      
    return pointsPhi,pointsG,phiout

def fg2dSpline(phiout,rout,G,PHI,R):
    """
    Function to interpolate the polarization angles and filter scales
    not-tabulated in the calibration functions stored.
    """
    #angles in radians
    #r as R/sigma
    g=1
    j=-1
    rExacto=False
    phiExacto=False
    if rout in R:
        j=np.where(np.array(R)==rout)[0][0]
        rExacto=True        
    else:
        notFound=True
        j=0
        while j<(len(R)-1) and notFound:
            if rout>R[j] and rout<R[j+1]:
                notFound=False
            else:
                j=j+1

        #I won't use r greater than the tabulated values in my code 
        #for values greater then the tabulated ones
        #I'll just use my biggest value of R/sigma without any interpolation
        if j==(len(R)-1):
             rExacto=True
            
    i=-1      
    if phiout in PHI[:,j]:
        i=np.where(np.array(PHI[:,j])==phiout)[0][0]
        phiExacto=True

    else:
        i=0
        notFound=True
        while i<(len(PHI[:,0])-1) and notFound: 
            if phiout<PHI[i+1,j] and phiout>PHI[i,j]:
                notFound=False
            else:   
                 i=i+1
    
        
    if phiExacto:
        if rExacto:
            g=G[i,j]
        else:
            pointsG=[]
            for s in np.arange(0,len(R),1):
                pointsG.append(G[i,s])
            
            cs=CubicSpline(R,pointsG)
            g=cs(rout)
    else:      
        if rExacto:
            pointsPhi,pointsG,phiAjuste=getPointsForInterpolation(i,j,phiout,G,PHI)
            cs=CubicSpline(pointsPhi,pointsG)
            g=cs(phiAjuste)
        else:
            interpolatedGPHI=[]
            for s in np.arange(0,len(R),1):
                pointsPhi,pointsG,phiAjuste=getPointsForInterpolation(i,s,phiout,G,PHI)
                cs=CubicSpline(pointsPhi,pointsG)
                interpolatedGPHI.append(cs(phiAjuste))
            
            cs=CubicSpline(R,interpolatedGPHI)
            g=cs(rout)
       
    return g,phiout

#limits of the dispersion ranges defining the different zones of the sky
zoneDefE=[[5.55,15.25],[0.64,1.75],[31.43,85.24]]
zoneDefB=[[1.82,4.78],[0.32,0.81],[17.81,45.93]]
#noise levels defined
noiseLevels=[[210,50,5],[120,50,5],[440,50,5]]

#read arguments from terminal call 
args=initial()
mapName=args.patchName[0]
nu=args.nu[0]
calibName=args.calibf_name[0]
k=args.noise_level[0]
verbose=args.v
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

sigma_noise=noiseLevels[n][k-1]
#the error is only known for one filter scale
R=1

#read fwhm
words3=calibName.split()
fwhmpix=float(words3[3])
nside=int(words3[1])
convFac=convert_units("uK_CMB","Jysr",nu)
sigma=fwhmpix/2.355482 
#load maps    
emap=np.load('patches/'+mapName+'/'+mapName+'_E.npy')   
bmap=np.load('patches/'+mapName+'/'+mapName+'_B.npy')    
if verbose:
    print('E and B patches correctly readed')

#load calibration functions
fE=np.load('config/E phi '+calibName+'.npy')
fB=np.load('config/B phi '+calibName+'.npy')
gE=np.load('config/E p '+calibName+'.npy')
gB=np.load('config/B p '+calibName+'.npy')
phiEcal=np.load('config/phi E '+calibName+'.npy')
phiBcal=np.load('config/phi B '+calibName+'.npy')
Rcal=np.arange(0.4,2.2,0.05)   

phiErrorBarE=np.load('config/error bar phi E.npy')
phiErrorBarB=np.load('config/error bar phi B.npy')
pErrorBarE=np.load('config/error bar p E.npy')
pErrorBarB=np.load('config/error bar p B.npy')

if verbose:
    print('Calibration functions and tabulated errors correctly readed')
    
lsize=int(nside/4)
centralpix=int(lsize/2)
apix=hp.nside2pixarea(nside)

#compute filter image in Fouries space
filtroxf=np.zeros((lsize,lsize),dtype=complex)
filtroyf=np.zeros((lsize,lsize),dtype=complex)
#coordinates origin in [centralpix,centralpix]
for i in np.arange(0, lsize,1, dtype=int):
    for j in np.arange(0,lsize,1, dtype=int):
        #x,y with sign of each quadrant
        #he cambiado i,j de orden para girar los ejes y salvar la diferencia
        #de angulos esfera-plano
        y=i-centralpix
        x=j-centralpix
        qx=x*(2*np.pi/lsize) 
        qy=y*(2*np.pi/lsize) 
        q2=qx**2+qy**2
        if x==0. and y==0.:
            theta=0.
            c=0.
        else:
            theta=np.arctan(y/x)
            if (x<0 and y>0) or (x<0 and y<0):
                theta=np.pi+theta
            elif x>0 and y<0:
                theta=2*np.pi+theta
            c=np.cos(2*theta)
                
            
        e=((R*sigma)**2*np.exp(-q2*(R*sigma)**2*0.5))/(2*np.pi)
        s=np.sin(2*theta)
        filtroxf[i,j]=c*e
        filtroyf[i,j]=s*e
    
#image to filter will already have a shift        
filtroX=ft.fftshift(filtroxf)
filtroY=ft.fftshift(filtroyf)  

for h in np.arange(0,len(emap),1):
    #filter the sources
    filMapFEx=np.zeros((lsize,lsize),dtype=complex)
    filMapFEy=np.zeros((lsize,lsize),dtype=complex)
    filMapFBx=np.zeros((lsize,lsize),dtype=complex)
    filMapFBy=np.zeros((lsize,lsize),dtype=complex)
    eF=ft.fft2(emap[h])  
    bF=ft.fft2(bmap[h]) 
    filMapFEx=np.array(eF,dtype=complex)*np.array(filtroX,dtype=complex)
    filMapFEy=np.array(eF,dtype=complex)*np.array(filtroY,dtype=complex)
    filMapFBx=np.array(bF,dtype=complex)*np.array(filtroX,dtype=complex)
    filMapFBy=np.array(bF,dtype=complex)*np.array(filtroY,dtype=complex)
    #polarization angle determination
    filMapREx=np.real(ft.ifft2(filMapFEx))
    filMapREy=np.real(ft.ifft2(filMapFEy))
    filMapRBx=np.real(ft.ifft2(filMapFBx))
    filMapRBy=np.real(ft.ifft2(filMapFBy))
    wxe=filMapREx[centralpix,centralpix]
    wye=filMapREy[centralpix,centralpix]
    wxb=filMapRBx[centralpix,centralpix]
    wyb=filMapRBy[centralpix,centralpix]
    phie=0.5*np.arctan(wye/wxe) 
    phib=0.5*np.arctan(-wxb/wyb)               
    #quadrant correction of arctan output                 
    if (wxe<0 and wye>0) or (wxe<0 and wye<0):
        phie=np.pi/2+phie
    	    
    if (wyb<0 and -wxb>0) or (wyb<0 and -wxb<0):
        phib=np.pi/2+phib
    	    
    
    #angle correction
    facEPhi,phiecal=fg2dSpline(phie,R,fE,phiEcal,Rcal)
    facBPhi,phibcal=fg2dSpline(phib,R,fB,phiBcal,Rcal)
    phiec=facEPhi*phiecal
    phibc=facBPhi*phibcal
    #give angles in the fourth quadrant their rightfull value
    if wxe>0 and wye<0:
        phief=np.rad2deg(phiec+np.pi)
    else:
        phief=np.rad2deg(phiec)
    if wyb>0 and -wxb<0:
        phibf=np.rad2deg(phibc+np.pi)
    else:
        phibf=np.rad2deg(phibc)
    
    we=np.cos(2*phie)*wxe+np.sin(2*phie)*wye
    wb=np.cos(2*phib)*wyb-np.sin(2*phib)*wxb
    pe=we*16*np.pi**2*(sigma**2+(R*sigma)**2)
    pb=wb*16*np.pi**2*(sigma**2+(R*sigma)**2)
    facEP,phiecal=fg2dSpline(phie,R,gE,phiEcal,Rcal)
    facBP,phibcal=fg2dSpline(phib,R,gB,phiBcal,Rcal)
    #recover the original intensity inptu
    pef=pe*facEP*apix*convFac
    pbf=pb*facBP*apix*convFac
    
    #identify to which zone the patch corresponds
    sigmaE=np.std(emap)
    sigmaB=np.std(bmap)
    #include noise in the zone definition
    zoneE=[np.sqrt(zoneDefE[n][0]**2+sigma_noise**2),np.sqrt(zoneDefE[n][1]**2+sigma_noise**2)]
    zoneB=[np.sqrt(zoneDefB[n][0]**2+sigma_noise**2),np.sqrt(zoneDefB[n][1]**2+sigma_noise**2)]
    
    if sigmaE<=zoneE[0]:
        ize=0
    elif zoneE[0]<sigmaE<=zoneE[1]:
        ize=1
    else:
        ize=2
        
    if sigmaB<=zoneB[0]:
        izb=0
    elif zoneB[0]<sigmaB<=zoneB[1]:
        izb=1
    else:
        izb=2
    
    ephie=phiErrorBarE[n][ize][k-1]
    ephib=phiErrorBarB[n][izb][k-1]
    epe=pErrorBarE[n][ize][k-1]
    epb=pErrorBarB[n][izb][k-1]
    
    print('    phiE erroPhiE   phiB erroPhiB     PE    errorPE     PB    errorPB')
    a=float("{0:3.4f}".format(phief))
    b=float("{0:3.4f}".format(phibf))
    c=float("{0:.4f}".format(pef))
    d=float("{0:.4f}".format(pbf))
    print(repr(a).rjust(7), repr(ephie).rjust(6), repr(b).rjust(9), repr(ephib).rjust(5), repr(c).rjust(11), repr(epe).rjust(5), repr(d).rjust(11), end=' ')
    print(repr(epb).rjust(4))
