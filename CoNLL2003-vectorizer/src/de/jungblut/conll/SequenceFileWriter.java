package de.jungblut.conll;

import java.io.*;
import java.util.List;
import java.util.StringJoiner;

import static com.google.common.base.Preconditions.checkArgument;

public interface SequenceFileWriter extends Closeable {

    /**
     * Writes the next example to the file.
     *
     * @param labels   the labels for the given sequence.
     * @param sequence the sequence vectors.
     * @throws IOException
     */
    public void write(int[] labels, List<float[]> sequence) throws IOException;

    public static class NoOpWriter implements SequenceFileWriter {

        @Override
        public void write(int[] labels, List<float[]> sequence) throws IOException {

        }

        @Override
        public void close() throws IOException {

        }
    }

    public static class TextWriter implements SequenceFileWriter {

        private final BufferedWriter writer;

        public TextWriter(String path) throws IOException {
            writer = new BufferedWriter(new FileWriter(path + ".txt"));
        }

        @Override
        public void write(int[] labels, List<float[]> sequence) throws IOException {
            checkArgument(labels.length == sequence.size(), "seq and feature size don't match");
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < labels.length; i++) {
                sb.append(labels[i]);
                if (i != labels.length - 1) {
                    sb.append(" ");
                }
            }

            for (float[] vector : sequence) {
                for (float d : vector) {
                    sb.append(" ");
                    sb.append(d);
                }
            }

            writer.write(sb.toString());
            writer.newLine();
        }

        @Override
        public void close() throws IOException {
            writer.close();
        }
    }

    public static class BinaryWriter implements SequenceFileWriter {

        private final DataOutputStream writer;

        public BinaryWriter(String path) throws IOException {
            writer = new DataOutputStream(new BufferedOutputStream(
                    new FileOutputStream(path + ".bin")));
        }

        @Override
        public void write(int[] labels, List<float[]> sequence) throws IOException {
            checkArgument(labels.length == sequence.size(), "seq and feature size don't match");
            writeVInt(writer, labels.length);
            for (int l : labels) {
                writeVInt(writer, l);
            }
            for (float[] vector : sequence) {
                for (float d : vector) {
                    writer.writeFloat(d);
                }
            }
        }

        @Override
        public void close() throws IOException {
            writer.close();
        }

        /**
         * Serializes an integer to a binary stream with zero-compressed encoding.
         * For -112 <= i <= 127, only one byte is used with the actual value. For
         * other values of i, the first byte value indicates whether the integer is
         * positive or negative, and the number of bytes that follow. If the first
         * byte value v is between -113 and -116, the following integer is positive,
         * with number of bytes that follow are -(v+112). If the first byte value v
         * is between -121 and -124, the following integer is negative, with number
         * of bytes that follow are -(v+120). Bytes are stored in the
         * high-non-zero-byte-first order.
         *
         * @param stream Binary output stream
         * @param i      Integer to be serialized
         * @throws java.io.IOException
         */
        public static void writeVInt(DataOutput stream, int i) throws IOException {
            writeVLong(stream, i);
        }

        /**
         * Serializes a long to a binary stream with zero-compressed encoding. For
         * -112 <= i <= 127, only one byte is used with the actual value. For other
         * values of i, the first byte value indicates whether the long is positive
         * or negative, and the number of bytes that follow. If the first byte value
         * v is between -113 and -120, the following long is positive, with number
         * of bytes that follow are -(v+112). If the first byte value v is between
         * -121 and -128, the following long is negative, with number of bytes that
         * follow are -(v+120). Bytes are stored in the high-non-zero-byte-first
         * order.
         *
         * @param stream Binary output stream
         * @param i      Long to be serialized
         * @throws java.io.IOException
         */
        public static void writeVLong(DataOutput stream, long i) throws IOException {
            if (i >= -112 && i <= 127) {
                stream.writeByte((byte) i);
                return;
            }

            int len = -112;
            if (i < 0) {
                i ^= -1L; // take one's complement'
                len = -120;
            }

            long tmp = i;
            while (tmp != 0) {
                tmp = tmp >> 8;
                len--;
            }

            stream.writeByte((byte) len);

            len = (len < -120) ? -(len + 120) : -(len + 112);

            for (int idx = len; idx != 0; idx--) {
                int shiftbits = (idx - 1) * 8;
                long mask = 0xFFL << shiftbits;
                stream.writeByte((byte) ((i & mask) >> shiftbits));
            }
        }

    }

}
