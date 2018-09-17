"""
Script to generate calibration functions for new combinations of nside and 
FHWM/pix ratios.

@author: Patricia Diego Palazuelos
"""
import argparse
import textwrap
import numpy as np
import healpy as hp
import scipy.fftpack as ft
from pysm.common import convert_units
import sys
import warnings
from scipy.interpolate import CubicSpline

"""
Ignore warnings to avoid printing the 'divide by zero' Runtime Warning when
polar coordinates are calculated for the source and filter images 
"""
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def initial():
    """
    Function to read input arguments from command line and to offer help to user.
    """
    parser = argparse.ArgumentParser(
    prog='newCalibrationFunction.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    newCalibrationFunction
    -------------
    Computes calibration functions for both polarization intensity and angle for new combinations of nside and FWHM/pix ratios.
    Default : dphi=1º dR/sigma=0.05.
    '''))
    parser.add_argument('FWHM_pix', metavar='fwhmpix', type=str, nargs=1,
                        help='Size of the FWHM of the Gaussian PSF in number of pixels. Several can be run as: "fwhm1 fwhm2 fwhm3 ... "')
    parser.add_argument('NSIDE', metavar='NSIDE', type=str, nargs=1,
                        help='HEALPIX NSIDE parameter describing sphere pixelization. Several can be run as: "NSIDE1 NSIDE2 NSIDE3 ... "')
    parser.add_argument('-r', metavar='dR/sigma',action='store', default=0.05,
                        help='Step in the R/sigma filter scales used to construct the function. Default=0.05')
    parser.add_argument('-a', metavar='dphi',action='store' , default=1,
                        help="Step in the angles used to construct the function. Default=1º")
    parser.add_argument('-v', action='store_true',
                        help='Verbose mode. Default = False')
    args      = parser.parse_args()
    return args

#read arguments from terminal call 
args=initial()
dphi=int(args.a)
dR=float(args.r)
verbose=args.v
fwhmstr=args.FWHM_pix[0]
NSIDE=args.NSIDE[0]
words1=fwhmstr.split()
words2=NSIDE.split()
fwhmpix=np.zeros_like(words1,dtype=float)
nside=np.zeros_like(words2,dtype=float)
for i in np.arange(0,len(words1),1):
    fwhmpix[i]=float(words1[i])
    nside[i]=int(words2[i])

#arbitrary parameters for source 
F=1#total flux in Jy 
nu=100.#frecuency in GHz
polDegree=1#polarization degree
phiStokes=np.arange(-44,136,dphi)#polarization angle 
Rcal=np.arange(0.4,2.2,dR)     
phi=np.zeros((len(phiStokes),len(Rcal)))
pointsInFunction=len(Rcal)*len(phiStokes)
for l in np.arange(0,len(phiStokes),1):
    for g in np.arange(0,len(Rcal),1):
        phi[l,g]=np.deg2rad(phiStokes[l])

for h in np.arange(0,len(fwhmpix),1):
    if verbose:
        print('FWHM '+str(h+1)+'/'+str(len(fwhmpix)))
    
    lsize=int(nside[h]/4)    
    centralpix=int(lsize/2)
    Apix=hp.nside2pixarea(nside[h])
    sigma=fwhmpix[h]/2.355482 
    R=np.array(Rcal)*sigma
    pE=np.zeros((len(phiStokes),len(R)),dtype=float)
    phiE=np.zeros((len(phiStokes),len(R)),dtype=float)
    phiB=np.zeros((len(phiStokes),len(R)),dtype=float)
    pB=np.zeros((len(phiStokes),len(R)),dtype=float)  
    #point-like sources
    sourcesE=[]
    sourcesB=[]
    Pi=(F*polDegree*convert_units("Jysr","uK_CMB",nu))/Apix  
    if verbose:
        print('Computing source images')
    
    progressCount=0   
    prevProgress=0
    for l in np.arange(0,len(phiStokes),1):
        sE=np.zeros((lsize,lsize),dtype=float)
        sB=np.zeros((lsize,lsize),dtype=float)
        #coordinates origin in [centralpix,centralpix]
        for i in np.arange(0, lsize,1, dtype=int):
            for j in np.arange(0,lsize,1, dtype=int):
                #x,y with sign of each quadrant
                y=i-centralpix
                x=j-centralpix
                r2=x**2+y**2
                if x==0. and y==0.:
                    theta=0.
                    c=0.
                    e=0.
                else:
                    theta=np.arctan(y/x)
                    if (x<0 and y>0) or (x<0 and y<0):
                        theta=np.pi+theta
                    elif x>0 and y<0:
                        theta=2*np.pi+theta
                    c=np.cos(2*theta)   
                    z=r2/(2*sigma**2)
                    e=(sigma**2*(np.exp(-z)*(1+z)-1))/(4*np.pi**2*r2)
                    
                
                s=np.sin(2*theta)
                sE[i,j]=(np.cos(2*np.deg2rad(phiStokes[l]))*c+np.sin(2*np.deg2rad(phiStokes[l]))*s)*e*Pi
                sB[i,j]=(np.cos(2*np.deg2rad(phiStokes[l]))*s-np.sin(2*np.deg2rad(phiStokes[l]))*c)*e*Pi
        
        sourcesE.append(sE) 
        sourcesB.append(sB)
        #progress bar        
        progressCount=progressCount+1
        if verbose:
            actualProgress=progressCount/len(phiStokes)
            if (actualProgress-prevProgress)>=0.1:
                prevProgress=actualProgress
                sys.stdout.write('\r')
                sys.stdout.write("[%-10s] %d%%" % ('='*int(actualProgress*10), actualProgress*100))
                sys.stdout.flush()
                
            if progressCount==pointsInFunction:
                sys.stdout.write('\r')
                sys.stdout.write("[%-10s] %d%%" % ('='*int(10), 100))
                sys.stdout.flush()
        
    
    if verbose:
        print('\nPoint-like source images computed\nProducing calibration functions')
    
    progressCount=0
    prevProgress=0   
    for g in np.arange(0,len(R),1):
        #filter image in fourier space
        filtroxf=np.zeros((lsize,lsize),dtype=complex)
        filtroyf=np.zeros((lsize,lsize),dtype=complex)
        #coordinates origin in [centralpix,centralpix]
        for i in np.arange(0, lsize,1, dtype=int):
            for j in np.arange(0,lsize,1, dtype=int):
                #x,y with sign of each quadrant
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
                        
                    
                e=(R[g]**2*np.exp(-q2*R[g]**2*0.5))/(2*np.pi)
                s=np.sin(2*theta)
                filtroxf[i,j]=c*e
                filtroyf[i,j]=s*e
            
        #image to filter will already have a shift        
        filtXf=ft.fftshift(filtroxf)
        filtYf=ft.fftshift(filtroyf)  
        
        #filtering
        for l in np.arange(0,len(phiStokes),1):
                filMapFEx=np.zeros((lsize,lsize),dtype=complex)
                filMapFEy=np.zeros((lsize,lsize),dtype=complex)
                filMapFBx=np.zeros((lsize,lsize),dtype=complex)
                filMapFBy=np.zeros((lsize,lsize),dtype=complex)
                eF=ft.fft2(sourcesE[l])  
                bF=ft.fft2(sourcesB[l]) 
                filMapFEx=np.array(eF,dtype=complex)*np.array(filtXf,dtype=complex)
                filMapFEy=np.array(eF,dtype=complex)*np.array(filtYf,dtype=complex)
                filMapFBx=np.array(bF,dtype=complex)*np.array(filtXf,dtype=complex)
                filMapFBy=np.array(bF,dtype=complex)*np.array(filtYf,dtype=complex)
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
    
                phiE[l,g]=phie
                phiB[l,g]=phib
                we=np.cos(2*phie)*wxe+np.sin(2*phie)*wye
                wb=np.cos(2*phib)*wyb-np.sin(2*phib)*wxb
                pE[l,g]=(we*16*np.pi**2*(sigma**2+R[g]**2))/(sigma*R[g])**2
                pB[l,g]=(wb*16*np.pi**2*(sigma**2+R[g]**2))/(sigma*R[g])**2
                
                #progress bar        
                progressCount=progressCount+1
                if verbose:
                    actualProgress=progressCount/pointsInFunction
                    if (actualProgress-prevProgress)>=0.1:
                        prevProgress=actualProgress
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-10s] %d%%" % ('='*int(actualProgress*10), actualProgress*100))
                        sys.stdout.flush()
                        
                    if progressCount==pointsInFunction:
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-10s] %d%%" % ('='*int(10), 100))
                        sys.stdout.flush()
                
        
        
    #calibration functions   
    gE=Pi/np.array(pE)
    gB=Pi/np.array(pB)
    fE=np.array(phi)/np.array(phiE)    
    fB=np.array(phi)/np.array(phiB)
    #correct discontinuity at phi=0 interpolating that point
    #search for phi=0 element index
    a=np.where(phiStokes==0)
    b,c=np.shape(a)
    if c==1:
        z=a[0][0]#index of element zero
        #limits for which point will be used for the interpolation
        lowerL=z-5
        upperL=z+7
        if lowerL<0:
            lowerL=lowerL+len(phiStokes)
        if upperL>(len(phiStokes)-1):
            upperL=upperL-len(phiStokes)
            
        for g in np.arange(0,len(Rcal),1):
            csE=CubicSpline(np.deg2rad(np.concatenate((phiStokes[lowerL:z],phiStokes[z+1:upperL]))), np.concatenate((fE[lowerL:z,g],fE[z+1:upperL,g])))
            csB=CubicSpline(np.deg2rad(np.concatenate((phiStokes[lowerL:z],phiStokes[z+1:upperL]))), np.concatenate((fB[lowerL:z,g],fB[z+1:upperL,g])))
            fE[z,g]=csE(0)
            fB[z,g]=csB(0)

                    
        
    #save results
    filenameEp='config/E p nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    filenameBp='config/B p nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    filenameEphi='config/E phi nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    filenameBphi='config/B phi nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    filenamephiE='config/phi E nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    filenamephiB='config/phi B nside '+str(nside[h])+' fw '+str(fwhmpix[h])+' dphi'+str(dphi*60)+' dR'+str(dR)
    
    np.save(filenameEphi,fE)
    np.save(filenameBphi,fB)
    np.save(filenamephiE,phiE)
    np.save(filenamephiB,phiB)
    np.save(filenameEp,gE)
    np.save(filenameBp,gB)
    
    if verbose:
        print('\nCalibration functions saved') 
 




