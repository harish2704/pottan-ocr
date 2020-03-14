const path = require('path');
const webpack = require('webpack');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

const config = {
  entry: [
    path.resolve(__dirname, 'js/app.pottan-tfjs.js'),
    path.resolve(__dirname, 'css/app.css'),
  ],
  resolve: { extensions: ['.js'] },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js',
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [[
              '@babel/preset-env', {
                targets: {
                  chrome: "77",
                },
              }
            ]],
            plugins: [
              '@babel/plugin-proposal-object-rest-spread'
            ],
          }
        }
      },
      {
        test: /\.css$/,
        loader: ExtractTextPlugin.extract({
          fallback: 'style-loader',
          use: ['css-loader']
        })
      },
    ]
  },
  node: {
    fs: 'empty'
  }
};

if (process.env.NODE_ENV === 'production') {
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    new webpack.optimize.ModuleConcatenationPlugin(),
    new ExtractTextPlugin({ filename: '[name].bundle.css', disable: false }),
  ];
  config.externals = {
    'jquery': 'jQuery',
    '@tensorflow/tfjs': 'tf',
    '@tensorflow/tfjs-vis': 'tfvis'
  };
} else {
  config.devtool = 'eval';
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') }),
    new ExtractTextPlugin({ filename: '[name].bundle.css', disable: false }),
  ];
}

module.exports = config;
