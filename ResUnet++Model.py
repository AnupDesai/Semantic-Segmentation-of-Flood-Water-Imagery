class ResUnetPlusPlus(pl.LightningModule):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_layer1 = Squeeze_block(filters[0])
        self.res_conv_layer1 = ResidualConvLayer(filters[0], filters[1], 2, 1)

        self.squeeze_layer2 = Squeeze_block(filters[1])
        self.res_conv_layer2 = ResidualConvLayer(filters[1], filters[2], 2, 1)

        self.squeeze_layer21 = Squeeze_block(filters[2])
        self.res_conv_layer21 = ResidualConvLayer(filters[2], filters[2], 2, 1)

        self.squeeze_layer22 = Squeeze_block(filters[2])
        self.res_conv_layer22 = ResidualConvLayer(filters[2], filters[2], 2, 1)

        self.squeeze_layer23 = Squeeze_block(filters[2])
        self.res_conv_layer23 = ResidualConvLayer(filters[2], filters[2], 2, 1)

        self.squeeze_layer24 = Squeeze_block(filters[2])
        self.res_conv_layer24 = ResidualConvLayer(filters[2], filters[2], 2, 1)
        
        self.squeeze_layer3 = Squeeze_block(filters[2])
        self.res_conv_layer3 = ResidualConvLayer(filters[2], filters[3], 2, 1)
        

        self.pyramid_bridge_layer = Pyramid_pooling(filters[3], filters[4])

        self.attn1 = Attention_unit_Block(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_res_conv_layer1 = ResidualConvLayer(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = Attention_unit_Block(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_res_conv_layer2 = ResidualConvLayer(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = Attention_unit_Block(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_res_conv_layer3 = ResidualConvLayer(filters[2] + filters[0], filters[1], 1, 1)

        self.pyramid_out_layer = Pyramid_pooling(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 2, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_layer1(x1)
        x2 = self.res_conv_layer1(x2)

        x3 = self.squeeze_layer2(x2)
        x3 = self.res_conv_layer2(x3)
        
        x4 = self.squeeze_layer3(x3)
        x4 = self.res_conv_layer3(x4)

        x5 = self.pyramid_bridge_layer(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_res_conv_layer1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_res_conv_layer2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_res_conv_layer3(x8)

        x9 = self.pyramid_out_layer(x8)
        out = self.output_layer(x9)

        return out
