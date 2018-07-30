import _ from 'lodash'
import unpack from 'ndarray-unpack'

/**
 * calculates mean and stddev for a ndarray tensor
 */
export function tensorStats(tensor) {
  const mean = _.sum(tensor.data) / tensor.data.length
  const stddev = Math.sqrt(_.sum(tensor.data.map(x => (x - mean) ** 2)) / tensor.data.length)
  return { mean, stddev }
}

/**
 * calculates min and max for a ndarray tensor
 */
export function tensorMinMax(tensor) {
  let min = Infinity
  let max = -Infinity
  for (let i = 0, len = tensor.data.length; i < len; i++) {
    if (tensor.data[i] < min) min = tensor.data[i]
    if (tensor.data[i] > max) max = tensor.data[i]
  }
  return { min, max }
}

/**
 * Takes in a ndarray of shape [x]
 * and creates image data
 */
export function image1Dtensor(tensor) {
  const { min, max } = tensorMinMax(tensor)
  let imageData = new Uint8ClampedArray(tensor.size * 4)
  for (let i = 0, len = imageData.length; i < len; i += 4) {
    imageData[i + 3] = 255 * (tensor.data[i / 4] - min) / (max - min)
  }
  return new ImageData(imageData, tensor.shape[0], 1)
}

/**
 * Takes in a ndarray of shape [x, y]
 * and creates image data
 */
export function image2Dtensor(tensor) {
  const { min, max } = tensorMinMax(tensor)
  // console.log( min, max );
  let imageData = new Uint8ClampedArray(tensor.size * 4)
  for (let i = 0, len = imageData.length; i < len; i += 4) {
    imageData[i + 3] = 255 * ( ( (min - tensor.data[i / 4]) / (max - min) ) + 1 )
  }
  return new ImageData(imageData, tensor.shape[1], tensor.shape[0])
}


/**
 * Takes in a ndarray of shape [x, y, z]
 * and creates an array of z ImageData [x, y] elements
 */
export function unroll3Dtensor(tensor) {
  const { min, max } = tensorMinMax(tensor)
  let shape = tensor.shape.slice()
  let unrolled = []
  for (let k = 0, channels = shape[2]; k < channels; k++) {
    const channelData = _.flatten(unpack(tensor.pick(null, null, k)))
    unrolled.push(channelData)
  }

  return unrolled.map(channelData => {
    let imageData = new Uint8ClampedArray(channelData.length * 4)
    for (let i = 0, len = channelData.length; i < len; i++) {
      imageData[i * 4] = 0
      imageData[i * 4 + 1] = 0
      imageData[i * 4 + 2] = 0
      imageData[i * 4 + 3] = 255 * (channelData[i] - min) / (max - min)
    }
    return new ImageData(imageData, shape[1], shape[0])
  })
}
