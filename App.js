import React, { useEffect, useState } from 'react';
import { View, Text, Button, Image, ScrollView, StyleSheet, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as cocossd from '@tensorflow-models/coco-ssd';
import * as jpeg from 'jpeg-js';
export default function App() {
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const initTensorFlow = async () => {
      await tf.ready();
    };
    initTensorFlow();
  }, []);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Permission to access gallery is required!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      detectObjects(result.assets[0].uri);
    }
  };
  const imageToTensor = (rawImageData) => {
    const { width, height, data } = jpeg.decode(rawImageData, { useTArray: true });
    const buffer = new Uint8Array(width * height * 3);

    let offset = 0;
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }

    return tf.tensor3d(buffer, [height, width, 3]);
  };
  const detectObjects = async (imageUri) => {
    try {
      await tf.ready();
      const model = await cocossd.load();

      const response = await fetch(imageUri, {}, { isBinary: true });
      const imageData = await response.arrayBuffer();
      const imageTensor = imageToTensor(new Uint8Array(imageData));

      const predictions = await model.detect(imageTensor);
      setPredictions(predictions);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Object Detection</Text>

      <Button title="Pick an image from gallery" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={styles.image} />}
      {predictions.length > 0 && (
        <View style={styles.predictionsContainer}>
          {predictions.map((prediction, index) => (
            <Text key={index} style={styles.predictionText}>
              {prediction.class} - {Math.round(prediction.score * 100)}%
            </Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  image: {
    width: 300,
    height: 300,
    marginVertical: 20,
  },
  predictionsContainer: {
    marginTop: 20,
  },
  predictionText: {
    fontSize: 16,
  },
});