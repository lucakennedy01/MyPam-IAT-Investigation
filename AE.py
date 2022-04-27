from create_dataset import *

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import matplotlib as mpl
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

epochs = int(config.get("autoencoder", "epochs"))
n = int(config.get("autoencoder", "reconstruction_dim")) #used in plot_reconstruction

device = 'cuda' if torch.cuda.is_available() else 'cpu' #gtx1060 doesn't support it :(

__spec__ = None

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(2000, 112) #http://dstath.users.uth.gr/papers/IJRS2009_Stathakis.pdf - Page 2 for hidden node calculation
        self.linear2 = nn.Linear(112, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 112)
        self.linear2 = nn.Linear(112, 2000)

    def forward(self, z):
        z = self.linear1(z)
        #print(z)
        z = F.relu(self.linear2(z))
        #print(z)
        return z.reshape((-1, 1, 1000, 2))

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=epochs):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print("Epoch {e} / {e_t} training...".format(e = epoch + 1, e_t = epochs))
        for x, y in data:
            #x = x.to(device) #if i can get CUDA 11.3
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
    return autoencoder

def plot_latent_space(autoencoder, data, num_batches=50):
    fig, ax = plt.subplots(1, 1, figsize = (7,5))

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    bounds = np.linspace(0,10,11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        scatter = ax.scatter(z[:, 0], z[:, 1], s=8, c=y, cmap=cmap, norm=norm)

    ax2 = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

    ax.set_title('Data Samples Encoded to 2D Latent Space ({E} Epochs, {B} Sampled Batches)'.format(E=epochs, B=num_batches), size = 10)
    ax2.set_ylabel('Dexterity [-]', size=7)
    #fig.tight_layout()
    plt.savefig('figures/latent_representations/LS_{E}-epochs_{B}-batches.png'.format(E=epochs, B=num_batches))

    plt.show()

def plot_reconstruction(autoencoder, r0 =(500, 560), r1= (370, 410), n=n):
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            print("Reconstructing with latent space X: {x}, Y: {y}".format(x=x, y=y))
            title = ("Reconstruction with Latent_X: {x}, Latent_Y: {y}".format(x=round(x,1),y=round(y,1)))
            z = torch.Tensor([[x,y]]).to(device)
            #print(z)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(1000, 2).to('cpu').detach().numpy()
            #print(len(x_hat))
            plt.scatter(x_hat[:,0], x_hat[:,1], s=6)
            plt.title(title)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig('figures/reconstructions/rec_x{x}_y{y}.png'.format(x=round(x,1),y=round(y,1)))
            plt.show()


def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    warnings.filterwarnings("ignore")

    latent_dims = 2
    autoencoder = Autoencoder(latent_dims)
    csv = get_labelled_file()
    transformed_dataset = transform_dataset(csv)

    #print("\nTransformed Dataset:\n")
    #for i in range(len(transformed_dataset)):
    #    sample = transformed_dataset[i]
    #    print(i, sample['label'], sample['points'].shape)
    #Map-Style Dataset
    data = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
    #print(type(data))
    #for i_batch, sample_batch in enumerate(data):
    #    if i_batch == 1:
    #        print(sample_batch)

    autoencoder = train(autoencoder, data)
    plot_latent_space(autoencoder, data)
    plot_reconstruction(autoencoder)


if __name__ == "__main__":
    main()
